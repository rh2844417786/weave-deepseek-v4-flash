"""Experimental SGLang 0.5.17 adapter for the functional persistent kernel.

Installation replaces every routed expert module. Unsupported paths raise;
there is no native expert fallback. Eager execution is the initial integration
target; graph replay of the core has its own GPU test and is not automatically
equivalent to SGLang's variable batch graph lifecycle.
"""
import copy
import importlib.metadata
import torch

from .ops import PeerRuntime
from .weights import load_raw_experts
from .runtime_contract import load_contract, verify_adapter_sources


class RoutedExperts(torch.nn.Module):
    def __init__(self, native, runtime, raw, profile, fixed_plan, clamp):
        super().__init__()
        self.runtime = runtime
        for j, tensor in enumerate(raw):
            self.register_buffer(f'raw_{j}', tensor)
        self.profile, self.fixed_plan, self.clamp = profile, fixed_plan, clamp
        self.quant_method = native.quant_method
        self.moe_runner_config = copy.copy(native.moe_runner_config)
        self.moe_runner_config.inplace = False
        self.should_fuse_routed_scaling_factor_in_topk = native.should_fuse_routed_scaling_factor_in_topk
        self.supports_deferred_finalize = False
        self.use_flashinfer_trtllm_moe = False
        self.coverage = {'prefill': 0, 'decode': 0, 'other': 0}

    def forward(self, hidden_states, topk_output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('SGLang graph lifecycle not validated; use explicit eager functional verification')
        if hasattr(topk_output, 'topk_ids') and hasattr(topk_output, 'topk_weights'):
            ids, weights = topk_output.topk_ids, topk_output.topk_weights
        elif hasattr(topk_output, 'packed_topk_ids'):
            from sglang.srt.layers.moe.moe_runner.marlin import _fused_unpack_packed_topk
            ids, weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
        else:
            raise ValueError('unsupported native routing output; no fallback')
        n = hidden_states.shape[0]
        chunk = (n + self.runtime.ep - 1) // self.runtime.ep
        start = min(n, self.runtime.rank * chunk)
        end = min(n, start + chunk)
        raw = tuple(getattr(self, f'raw_{j}') for j in range(6))
        own = self.runtime.run(hidden_states[start:end], ids[start:end], weights[start:end],
                               raw, profile=self.profile, fixed_plan=self.fixed_plan, clamp=self.clamp)
        # Attention TP has replicated token rows. Each origin returns its owned
        # rows only; the native outer all-reduce restores replicated routed output.
        partial = torch.zeros_like(hidden_states)
        partial[start:end].copy_(own)
        self.coverage[self.runtime.phase] += 1
        return partial

    def forward_impl(self, hidden_states, topk_output):
        return self.forward(hidden_states, topk_output)

    def forward_deferred_finalize(self, *args, **kwargs):
        raise RuntimeError('deferred native finalize is outside this adapter contract')


def install_weave_backend(model, model_path, *, capacity=2048, profile_path=None, fixed_plan=(1, 1, 0)):
    expected_version = load_contract()['sglang_version']
    if importlib.metadata.version('sglang') != expected_version:
        raise RuntimeError(f'adapter requires the source-audited SGLang {expected_version}')
    verify_adapter_sources()
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel, get_server_args
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('install the adapter before CUDA Graph capture')
    args, parallel, config = get_server_args(), get_parallel(), model.config
    if (parallel.tp_size != 4 or parallel.moe_ep_size != 4 or parallel.moe_tp_size != 1
            or parallel.attn_dp_size != 1 or getattr(args, 'pp_size', 1) != 1
            or getattr(parallel, 'attn_cp_size', 1) != 1
            or getattr(args, 'enable_eplb', False) or getattr(args, 'speculative_algorithm', None)
            or getattr(args, 'enable_torch_compile', False) or is_sbo_enabled() or is_tbo_enabled()):
        raise ValueError('functional adapter requires TP4/EP4, expert TP1, attention DP1, no PP/EPLB/MTP')
    if getattr(args, 'moe_a2a_backend', 'none') != 'none':
        raise ValueError('adapter owns dispatch/combine and requires the standard outer runtime path')
    if (config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            config.num_experts_per_tok, config.num_hidden_layers) != (4096, 2048, 256, 6, 43):
        raise ValueError('unexpected DeepSeek-V4-Flash architecture')
    layers = list(model.model.layers)
    if len(layers) != 43:
        raise ValueError('all 43 layers must be present on every rank')
    # Validate every replacement before allocating/loading new weights.
    for layer in layers:
        mlp = layer.mlp
        if (not isinstance(mlp.experts.quant_method, Mxfp4MarlinMoEMethod)
                or mlp.num_fused_shared_experts != 0 or mlp._enable_a2a_moe
                or mlp.experts.should_fuse_routed_scaling_factor_in_topk
                or getattr(mlp, '_fuse_shared_experts_inside_sbo', False)):
            raise ValueError('unsupported expert/scaling/shared-expert path; no partial fallback')
    runtime = PeerRuntime(get_tp_group().cpu_group, capacity, 4096, 2048, 6, 256)
    profile = runtime.profile_from_file(profile_path) if profile_path else None
    for layer in layers:
        raw = load_raw_experts(model_path, layer.mlp.layer_id, runtime.rank, runtime.ep,
                               device=torch.device('cuda', runtime.device))
        layer.mlp.experts = RoutedExperts(layer.mlp.experts, runtime, raw, profile, fixed_plan, config.swiglu_limit)
    model.weave_runtime = runtime
    return {'backend': 'weave_functional', 'layers': len(layers), 'graph_status': 'not_validated',
            'schedule_mode': 'measured_dynamic' if profile is not None else 'explicit_fixed'}


def coverage(model):
    return {str(layer.mlp.layer_id): dict(layer.mlp.experts.coverage) for layer in model.model.layers}
