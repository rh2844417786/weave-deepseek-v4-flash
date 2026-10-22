"""Private native Marlin front and explicit decode-only MoE tail boundary.

The front is derived from hash-verified installed SGLang source, stopping before
GEMM2. No installed module function or global collective is replaced. The native
gate/up, grouping and activation expressions execute unchanged. These Python
objects keep the graph-captured tensors alive; no replay reads routing on CPU.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

from .native_tail_patch import source_pins

FUSED_SOURCE = 'srt/layers/moe/fused_moe_triton/fused_marlin_moe.py'
RUNNER_SOURCE = 'srt/layers/moe/moe_runner/marlin.py'


@dataclass(frozen=True)
class NativeGeometry:
    tokens: int
    topk: int
    pair_rows: int
    n: int
    k: int
    block_size_m: int
    local_experts: int
    group_capacity: int


@dataclass(frozen=True)
class DownInputs:
    activated_pairs: Any
    sorted_token_ids: Any
    local_expert_ids: Any
    num_tokens_post_padded: Any
    topk_weights: Any
    topk_ids: Any
    original_slot_mapping: Any
    w2: Any
    w2_scales: Any
    geometry: NativeGeometry
    # Strong references include native scratch and the original input. Capture
    # owners retain this bundle through graph destruction, not just launch.
    owners: tuple[Any, ...]


def _verified_function(source, relative_path, function_name):
    if hashlib.sha256(source.encode()).hexdigest() != source_pins()[relative_path]:
        raise RuntimeError('native tail source mismatch: ' + relative_path)
    found = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == function_name]
    if len(found) != 1:
        raise RuntimeError('native tail extraction function differs: ' + function_name)
    function = found[0]
    function.decorator_list = []
    # Postponed evaluation lets this source-only loader avoid importing torch.
    return function


def _compile_function(function, namespace, filename):
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace[function.name]


def _down_inputs(values):
    return DownInputs(
        activated_pairs=values['intermediate_cache2'],
        sorted_token_ids=values['sorted_token_ids'],
        local_expert_ids=values['expert_ids'],
        num_tokens_post_padded=values['num_tokens_post_padded'],
        topk_weights=values['topk_weights'], topk_ids=values['topk_ids'],
        # sorted_token_ids is the original flattened (token, top-k slot) map;
        # grouping must never be mistaken for a dense local-slot permutation.
        original_slot_mapping=values['sorted_token_ids'],
        w2=values['w2'], w2_scales=values['w2_scale'],
        geometry=NativeGeometry(values['M'], values['topk'],
            values['M'] * values['topk'], values['K'], values['N'],
            values['block_size_m'], values['E'], min(values['expert_ids'].numel(),
                values['sorted_token_ids'].numel() // values['block_size_m'])),
        owners=tuple(values[k] for k in ('hidden_states', 'w1', 'w1_scale',
            'gating_output', 'workspace', 'intermediate_cache13', 'expert_map')),
    )


def build_front(source, native_module):
    """Build a callable with the original fused_marlin_moe argument signature."""
    function = _verified_function(source, FUSED_SOURCE, 'fused_marlin_moe')
    down_indexes = [i for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id == 'intermediate_cache3' for t in n.targets)
        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == 'moe_wna16_marlin_gemm' for c in ast.walk(n))]
    if len(down_indexes) != 1:
        raise RuntimeError('native tail extraction must find one native down')
    index = down_indexes[0]
    # The preceding zero_ only clears the aliased GEMM2 output, after activation.
    previous = function.body[index - 1]
    if (not isinstance(previous, ast.If) or ast.unparse(previous.test) != 'expert_map is not None'
            or ast.unparse(previous.body[0]) != 'intermediate_cache3.zero_()'):
        raise RuntimeError('native tail extraction down boundary differs')
    function.body = function.body[:index - 1] + ast.parse('return _weave_down_inputs(locals())').body
    function.name = 'native_marlin_front'
    namespace = dict(vars(native_module), _weave_down_inputs=_down_inputs)
    derived = _compile_function(function, namespace, '<weave-native-marlin-front>')
    signature = inspect.signature(derived)
    torch = native_module.torch

    @wraps(derived)
    def checked(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        if (p['hidden_states'].dtype != torch.bfloat16 or p['num_bits'] != 4
                or p['w1_scale'].dtype != torch.float8_e8m0fnu
                or p['w2_scale'].dtype != torch.float8_e8m0fnu):
            raise RuntimeError('native tail requires BF16 activations and native MXFP4/E8M0 weights')
        if (not p['is_k_full'] or p['expert_map'] is None or p['inplace']
                or any(p[k] is not None for k in ('g_idx1', 'g_idx2', 'sort_indices1',
                    'sort_indices2', 'w1_zeros', 'w2_zeros', 'w1_global_scale',
                    'w2_global_scale', 'w1_bias', 'w2_bias'))):
            raise RuntimeError('native tail requires EP, full K, no bias/zero/order/global-scale and distinct output')
        for name in ('w1', 'w2', 'topk_ids'):
            if p[name].dtype != torch.int32 or not p[name].is_contiguous():
                raise RuntimeError('native tail requires contiguous int32 ' + name)
        if p['topk_weights'].dtype != torch.float32 or not p['topk_weights'].is_contiguous():
            raise RuntimeError('native tail requires contiguous float32 top-k weights')
        if p['topk_ids'].shape != p['topk_weights'].shape or p['topk_ids'].shape[0] != p['hidden_states'].shape[0]:
            raise RuntimeError('native tail malformed top-k shape')
        return derived(*args, **kwargs)

    checked.weave_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return checked


def build_runner_front(source, native_module, front):
    """Preserve native runner argument packing, but return DownInputs, not output."""
    function = _verified_function(source, RUNNER_SOURCE, 'fused_experts_none_to_marlin')
    function.name = 'native_marlin_runner_front'
    body = []
    output_found = False
    for node in function.body:
        if isinstance(node, ast.ImportFrom) and node.module in (
            'sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe',
            'sglang.srt.layers.moe.token_dispatcher.standard'):
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'output' for t in node.targets):
            # Native runner assigns fused_marlin_moe(...).to(input_dtype).
            outer = node.value
            if (not isinstance(outer, ast.Call) or not isinstance(outer.func, ast.Attribute)
                    or outer.func.attr != 'to' or not isinstance(outer.func.value, ast.Call)
                    or not isinstance(outer.func.value.func, ast.Name)
                    or outer.func.value.func.id != 'fused_marlin_moe'):
                raise RuntimeError('native tail runner call anchor differs')
            body.append(ast.Return(value=outer.func.value))
            output_found = True
            break
        body.append(node)
    if not output_found:
        raise RuntimeError('native tail runner output anchor missing')
    function.body = body
    return _compile_function(function, dict(vars(native_module), fused_marlin_moe=front),
                             '<weave-native-marlin-runner-front>')


def load_native_front(package_root):
    root = Path(package_root)
    native = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')
    runner = importlib.import_module('sglang.srt.layers.moe.moe_runner.marlin')
    front = build_front((root / FUSED_SOURCE).read_text(), native)
    return build_runner_front((root / RUNNER_SOURCE).read_text(), runner, front)


class NativeDecodeFront:
    """Explicit DeepseekV2MoE decode helper for the verified TP4 shared layout.

    ``tail(down, shared, *, layer_id, routed_scaling_factor, output_width)`` must
    return the complete globally reduced output. It owns pair zeroing, down,
    local finalize and the one MoE output collective. Native Marlin writes
    top-k-weighted BF16 pairs; native topk_sum.cuh sums slots in ascending order
    in FP32 then rounds to BF16. ``maybe_fuse_routed_scale_and_shared_add`` then
    performs shared.add_(routed, alpha=routed_scaling_factor). The tail must
    preserve both rounding points and apply that outer scale exactly once.

    The caller installs this only for decode. Prefill uses its original bound
    forward, and attention/other collectives never pass through this helper.
    """
    def __init__(self, torch, native_module, runner_front, build_quant_info,
                 standard_dispatcher_type, quant_method_type):
        self.torch, self.native = torch, native_module
        self.runner_front, self.build_quant_info = runner_front, build_quant_info
        self.standard_dispatcher_type = standard_dispatcher_type
        self.quant_method_type = quant_method_type

    def validate_layer(self, mlp):
        experts = mlp.experts
        if mlp._shared_expert_tp1:
            raise RuntimeError('native tail does not support shared TP1')
        if (mlp.tp_size != 4 or mlp.moe_ep_size != 4 or mlp._enable_a2a_moe
                or mlp.num_fused_shared_experts != 0 or not mlp.shared_experts_is_fp8
                or not hasattr(mlp, 'shared_experts') or mlp.alt_stream is None
                or mlp._fuse_shared_experts_inside_sbo):
            raise RuntimeError('native tail requires TP4/EP4, separate FP8 shared and explicit alternate stream')
        if (not isinstance(experts.dispatcher, self.standard_dispatcher_type)
                or not isinstance(experts.quant_method, self.quant_method_type)
                or experts.dispatcher.skip_local_expert_mapping
                or experts.reduce_results or getattr(experts, '_dwdp_bound', False)):
            raise RuntimeError('native tail requires native MXFP4 Marlin standard EP mapping without internal reduction or weight prefetch')
        config = experts.moe_runner_config
        if (config.apply_router_weight_on_input
                or config.activation != 'silu' or not config.is_gated
                or experts.runner.lora_enabled or not experts.runner.runner_backend.is_marlin()):
            raise RuntimeError('native tail unsupported native runner/activation/LoRA configuration')
        topk_config = getattr(mlp.topk, 'topk_config', mlp.topk)
        if (experts.should_fuse_routed_scaling_factor_in_topk
                or getattr(topk_config, 'apply_routed_scaling_factor_on_output', False)):
            raise RuntimeError('native tail requires the native outer routed scaling location')
    def __call__(self, mlp, hidden_states, tail, *, forward_batch=None,
                 gemm_output_zero_allocator=None, input_ids=None,
                 input_ids_global=None, skip_shared_experts=False, observer=None):
        self.validate_layer(mlp)
        if hidden_states.dtype != self.torch.bfloat16:
            raise RuntimeError('native tail decode requires the BF16 model activation contract')
        if skip_shared_experts:
            raise RuntimeError('native tail requires the TP4 shared partial at tail entry')
        if mlp.is_hash and input_ids_global is None:
            raise RuntimeError('native tail hash routing requires original input_ids_global')
        # These flags are scoped to the current forward. Runtime construction
        # validates static layout only, before any forward context exists.
        if self.native.get_forward().flashinfer_trtllm_bypass:
            raise RuntimeError('native tail does not support bypassed routing')
        if self.native.should_skip_post_experts_all_reduce(is_tp_path=True):
            raise RuntimeError('native tail cannot replace a skipped/fused/reduce-scatter MoE collective')
        if observer is not None:
            observer.mark('entry')
        current_stream = self.torch.cuda.current_stream()
        pre_quant_input = mlp._maybe_quant_moe_input_once(hidden_states)
        # Source-verified native Marlin is ineligible for quantize-once, whose
        # eligibility is restricted to the Triton FP8 routed runner.
        if pre_quant_input is not None:
            raise RuntimeError('native tail cannot consume a prequantized routed activation')
        mlp.alt_stream.wait_stream(current_stream)
        dispatch_info = (self.native.ExpertLocationDispatchInfo.init_new(layer_id=mlp.layer_id)
            if self.native.get_exec().moe.enable_eplb and not mlp.is_nextn else None)
        if hidden_states.shape[0] > 0:
            router_logits = mlp.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = {'input_ids': input_ids_global} if mlp.is_hash else {}
            topk_output = mlp.topk(hidden_states, router_logits,
                expert_location_dispatch_info=dispatch_info, **topk_kwargs)
        else:
            topk_output = mlp.topk.empty_topk_output(hidden_states.device, layer_id=mlp.layer_id)
        # This is the original dispatcher: global -> local id remapping, remote
        # -1 sentinels and DP padding masking all happen in native code.
        dispatched = mlp.experts.dispatcher.dispatch(hidden_states=hidden_states, topk_output=topk_output)
        if hasattr(dispatched, 'format') and not dispatched.format.is_standard():
            raise RuntimeError('native tail needs standard native dispatch output')
        if not hasattr(dispatched.topk_output, 'topk_ids'):
            raise RuntimeError('native tail requires standard top-k before EP mapping')
        target_width = mlp.experts.w13_weight.shape[1] * 16
        if hidden_states.shape[1] > target_width:
            raise RuntimeError('native tail checkpoint hidden geometry is smaller than input')
        if dispatched.hidden_states.shape[1] != target_width:
            padded = self.torch.nn.functional.pad(dispatched.hidden_states,
                (0, target_width - dispatched.hidden_states.shape[1]), mode='constant', value=0.)
            dispatched = dispatched._replace(hidden_states=padded)
        down = self.runner_front(dispatched, self.build_quant_info(mlp.experts),
                                 mlp.experts.moe_runner_config)
        # Match native dual-stream issue order, but join immediately after the
        # native front. No persistent CTA can wait on an unlaunched shared op.
        with self.torch.cuda.stream(mlp.alt_stream):
            shared_output = mlp._forward_shared_experts(hidden_states,
                gemm_output_zero_allocator, pre_quant_input=pre_quant_input)
        if observer is not None:
            observer.mark('before_join')
        current_stream.wait_stream(mlp.alt_stream)
        if observer is not None:
            observer.mark('after_join')
            observer.mark('tail_entry')
        # The tail owns the shared addition and global reduction. Returning
        # here bypasses only this layer's former outer add/all-reduce.
        return tail(down, shared_output, layer_id=mlp.layer_id,
            routed_scaling_factor=mlp.routed_scaling_factor,
            output_width=hidden_states.shape[1])


def load_decode_front(torch, package_root):
    native = importlib.import_module('sglang.srt.models.deepseek_v2')
    quant = importlib.import_module('sglang.srt.layers.quantization.mxfp4_marlin_moe')
    dispatcher = importlib.import_module('sglang.srt.layers.moe.token_dispatcher.standard')
    return NativeDecodeFront(torch, native, load_native_front(package_root),
        quant.build_marlin_moe_quant_info, dispatcher.StandardDispatcher, quant.Mxfp4MarlinMoEMethod)
