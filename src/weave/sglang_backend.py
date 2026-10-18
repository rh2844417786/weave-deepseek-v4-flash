"""Experimental SGLang 0.5.17 adapter for the functional persistent kernel.

Installation replaces every routed expert module. Unsupported paths raise;
there is no native expert fallback. Eager execution is the initial integration
target; graph replay of the core has its own GPU test and is not automatically
equivalent to SGLang's variable batch graph lifecycle.
"""
import copy
import os
import importlib.metadata
from dataclasses import asdict
import torch

from .ops import PeerRuntime
from .weights import load_raw_experts
from .runtime_contract import load_contract, verify_adapter_sources
from .staged_decode import StagedDecodeRuntime, ModelBatchCounters, validate_decode_configuration, validate_prefill_configuration
from .hopper_decode import HopperDecodeRuntime
from .hopper_prefill import HopperPrefillRuntime


class RoutedExperts(torch.nn.Module):
    def __init__(self, native, runtime, raw, profile, fixed_plan, clamp,layer_id):
        super().__init__()
        self.runtime = runtime
        for j, tensor in enumerate(raw):
            self.register_buffer(f'raw_{j}', tensor)
        self.raw_weights = tuple(raw)
        # Installation happens before any graph capture. The bound owner keeps
        # all six tensors alive and removes per-layer static host inspection.
        self.bound_weights = runtime.bind_weights(self.raw_weights) if runtime.hot_cache else None
        self.staged_bound_weights = (runtime.staged_decode.bind_weights(self.raw_weights)
                                     if runtime.staged_decode is not None else None)
        hopper = getattr(runtime, 'hopper_decode', None)
        self.hopper_bound_weights = hopper.bind_weights(self.raw_weights) if hopper is not None else None
        if hopper is not None and getattr(runtime, 'graph_stage_profile', False):
            hopper.register_layer(layer_id, self.hopper_bound_weights)
        prefill = getattr(runtime, 'hopper_prefill', None)
        self.hopper_prefill_bound_weights = prefill.bind_weights(self.raw_weights) if prefill is not None else None
        self.profile, self.fixed_plan, self.clamp = profile, fixed_plan, clamp
        self.layer_id=layer_id
        self.quant_method = native.quant_method
        self.moe_runner_config = copy.copy(native.moe_runner_config)
        self.moe_runner_config.inplace = False
        self.should_fuse_routed_scaling_factor_in_topk = native.should_fuse_routed_scaling_factor_in_topk
        self.supports_deferred_finalize = False
        self.use_flashinfer_trtllm_moe = False
        self.coverage = {'prefill': 0, 'decode': 0, 'other': 0}
        self.implementation_calls = {'staged_decode': 0, 'persistent_decode': 0,
                                     'persistent_prefill': 0, 'persistent_other': 0}
        if hopper is not None:
            self.implementation_calls['hopper_decode'] = 0
        if prefill is not None:
            self.implementation_calls['hopper_prefill'] = 0
        self.capture_calls = 0
        self.register_buffer('device_calls', torch.zeros(3, dtype=torch.int64, device=f'cuda:{runtime.device}'))

    def forward(self, hidden_states, topk_output):
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and (not self.runtime.allow_decode_graph or self.runtime.phase != 'decode'
                          or (not getattr(self.runtime, 'graph_c32', False) and hidden_states.shape[0] != 1)):
            raise RuntimeError('graph candidate supports exactly one decode row and requires explicit opt-in')
        if hasattr(topk_output, 'topk_ids') and hasattr(topk_output, 'topk_weights'):
            ids, weights = topk_output.topk_ids, topk_output.topk_weights
        elif hasattr(topk_output, 'packed_topk_ids'):
            from sglang.srt.layers.moe.moe_runner.marlin import _fused_unpack_packed_topk
            ids, weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
        else:
            raise ValueError('unsupported native routing output; no fallback')
        n = hidden_states.shape[0]
        replay_capture = getattr(self.runtime, 'replay_capture', None)
        if replay_capture is not None:
            replay_capture.record_layer(self.layer_id, hidden_states, ids, weights)
        phase = self.runtime.phase
        if phase == 'prefill' and getattr(self.runtime, 'prefill_implementation', 'persistent') == 'hopper_fp8':
            partial = self.runtime.hopper_prefill.run(hidden_states, ids, weights,
                self.hopper_prefill_bound_weights, self.clamp)
            implementation = 'hopper_prefill'
        elif phase == 'decode' and self.runtime.decode_implementation == 'hopper_fp8':
            # Full replicated input; the Hopper finalizer already masks origins.
            # SGLang still owns shared experts, scaling and the outer TP SUM.
            partial = self.runtime.hopper_decode.run(hidden_states, ids, weights,
                                                     self.hopper_bound_weights, self.clamp)
            implementation = 'hopper_decode'
        elif phase == 'decode' and self.runtime.decode_implementation == 'staged_direct':
            # The staged workspace consumes full replicated routing and returns
            # an already origin-masked BF16 result after its slot SUM. The native
            # outer shared/scaling/TP reduction is still applied by SGLang.
            partial = self.runtime.staged_decode.run(hidden_states, ids, weights,
                                                     self.staged_bound_weights, self.clamp)
            implementation = 'staged_decode'
        else:
            chunk = (n + self.runtime.ep - 1) // self.runtime.ep
            start = min(n, self.runtime.rank * chunk)
            end = min(n, start + chunk)
            raw = self.bound_weights if self.bound_weights is not None else self.raw_weights
            own = self.runtime.run(hidden_states[start:end], ids[start:end], weights[start:end],
                                   raw, profile=self.profile, fixed_plan=self.fixed_plan, clamp=self.clamp,
                                   trace_layer=self.layer_id)
            # Persistent prefill and the persistent decode control retain the
            # existing contiguous origin partition and outer TP restoration.
            partial = torch.zeros_like(hidden_states)
            partial[start:end].copy_(own)
            implementation = 'persistent_' + phase
        self.device_calls[{'prefill': 0, 'decode': 1, 'other': 2}[phase]].add_(1)
        if capturing:
            self.capture_calls += 1
        else:
            self.coverage[phase] += 1
            self.implementation_calls[implementation] += 1
        return partial

    def forward_impl(self, hidden_states, topk_output):
        return self.forward(hidden_states, topk_output)

    def forward_deferred_finalize(self, *args, **kwargs):
        raise RuntimeError('deferred native finalize is outside this adapter contract')


def install_weave_backend(model, model_path, *, capacity=2048, profile_path=None, fixed_plan=(1, 1, 0),
                          kernel='scalar', allow_decode_graph=False,execution_profile=None,
                          decode_implementation='persistent', prefill_implementation='persistent'):
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
    graph_c32 = os.environ.get('WEAVE_GRAPH_C32') == '1'
    from .graph_observer import graph_stage_options
    stage_options = graph_stage_options()
    stage_profiler = getattr(model, 'weave_graph_stage_profiler', None)
    if stage_options is not None:
        if (not graph_c32 or not allow_decode_graph or decode_implementation != 'hopper_fp8'
                or prefill_implementation != 'hopper_fp8'):
            raise ValueError('graph stage profiling requires full C32 Hopper decode and prefill')
        if stage_profiler is None or stage_profiler.kind != 'weave':
            raise ValueError('graph stage profiler must be installed by the full graph runner before adapter warmup')
    if graph_c32:
        from .graph_observer import validate_graph_configuration
        validate_graph_configuration(args, parallel)
    if allow_decode_graph and not graph_c32 and (getattr(args, 'max_running_requests', None) != 1
                               or list(args.cuda_graph_config.decode.bs) != [1]):
        raise ValueError('graph candidate requires max-running-requests=1 and decode graph buckets=[1]')
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
    hot_cache_value=os.environ.get('WEAVE_HOT_CACHE','1')
    if hot_cache_value not in ('0','1'):raise ValueError('WEAVE_HOT_CACHE must be 0 or 1')
    hot_cache=hot_cache_value=='1'
    validate_decode_configuration(decode_implementation=decode_implementation,kernel=kernel,
        hot_cache=hot_cache,allow_decode_graph=allow_decode_graph,fixed_plan=fixed_plan,
        profile_path=profile_path,execution_profile=execution_profile,
        replay_capture=bool(os.environ.get('WEAVE_REPLAY_CAPTURE_DIR')),
        forward_profile=os.environ.get('WEAVE_FORWARD_PROFILE')=='1', graph_c32=graph_c32)
    validate_prefill_configuration(prefill_implementation=prefill_implementation,
        decode_implementation=decode_implementation,kernel=kernel,hot_cache=hot_cache,
        allow_decode_graph=allow_decode_graph,fixed_plan=fixed_plan,
        profile_path=profile_path,execution_profile=execution_profile,
        replay_capture=bool(os.environ.get('WEAVE_REPLAY_CAPTURE_DIR')),
        forward_profile=os.environ.get('WEAVE_FORWARD_PROFILE')=='1', graph_c32=graph_c32)
    runtime = PeerRuntime(get_tp_group().cpu_group, capacity, 4096, 2048, 6, 256,
                          kernel=kernel, cache_views=hot_cache)
    runtime.graph_c32 = graph_c32
    runtime.graph_stage_profile = stage_options is not None
    runtime.hot_cache=hot_cache
    runtime.allow_decode_graph = allow_decode_graph
    runtime.fixed_plan=tuple(fixed_plan)
    runtime.decode_implementation=decode_implementation
    runtime.staged_decode=(StagedDecodeRuntime(runtime,get_tp_group().device_group)
                           if decode_implementation=='staged_direct' else None)
    if graph_c32:
        from .hopper_graph import HopperGraphDecodeRuntime
    hopper_runtime_class = HopperGraphDecodeRuntime if graph_c32 else HopperDecodeRuntime
    if stage_options is not None:
        from .graph_stage_profile import HopperStageDecodeRuntime
        runtime.hopper_decode = HopperStageDecodeRuntime(runtime, get_tp_group().device_group, stage_profiler)
    else:
        runtime.hopper_decode=(hopper_runtime_class(runtime,get_tp_group().device_group)
                               if decode_implementation=='hopper_fp8' else None)
    runtime.prefill_implementation=prefill_implementation
    runtime.hopper_prefill=(HopperPrefillRuntime(runtime,get_tp_group().device_group)
                            if prefill_implementation=='hopper_fp8' else None)
    runtime.model_batch_counters=ModelBatchCounters()
    profile = runtime.profile_from_file(profile_path) if profile_path else None
    if execution_profile is not None: runtime.enable_execution_profile(execution_profile)
    if os.environ.get('WEAVE_FORWARD_PROFILE')=='1':
        if execution_profile is None or allow_decode_graph or profile is not None:
            raise ValueError('full-forward diagnostics require profiled fixed/eager execution')
        from .forward_profile import ForwardSampler
        runtime.forward_sampler=ForwardSampler(torch.cuda,stride=execution_profile.stride,
                                               capacity=execution_profile.capacity)
    if os.environ.get('WEAVE_REPLAY_CAPTURE_DIR'):
        if (kernel!='tiled' or allow_decode_graph or profile is not None or execution_profile is not None
                or tuple(fixed_plan)!=(1,1,0) or getattr(args,'max_running_requests',None)!=32):
            raise ValueError('real replay capture requires tiled C32 fixed(1,1,0), eager and no profiling')
        from .replay_trace import ReplayCapture,capture_source_sha256,tokenizer_manifest_sha256
        tokenizer_sha256=tokenizer_manifest_sha256(model_path)
        if tokenizer_sha256!=os.environ.get('WEAVE_REPLAY_TOKENIZER_MANIFEST_SHA256'):
            raise ValueError('local capture tokenizer/config/encoding assets differ from launcher identity')
        runtime.replay_capture=ReplayCapture(torch,runtime.device,runtime.rank,
            os.environ['WEAVE_REPLAY_CAPTURE_DIR'],{
                'kernel_source_sha256':runtime.source_sha256,
                'checkpoint_content_sha256':os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256'),
                'dataset_sha256':os.environ.get('WEAVE_REPLAY_DATASET_SHA256'),
                'tokenizer_manifest_sha256':tokenizer_sha256,
                'capture_id':os.environ.get('WEAVE_REPLAY_CAPTURE_ID'),
                'capture_source_sha256':capture_source_sha256(),
                'image_id':os.environ.get('WEAVE_IMAGE_ID')})
    for layer in layers:
        raw = load_raw_experts(model_path, layer.mlp.layer_id, runtime.rank, runtime.ep,
                               device=torch.device('cuda', runtime.device))
        layer.mlp.experts = RoutedExperts(layer.mlp.experts, runtime, raw, profile, fixed_plan, config.swiglu_limit,layer.mlp.layer_id)
    model.weave_runtime = runtime
    return {'backend': 'weave_functional', 'kernel': kernel, 'layers': len(layers),
            'graph_status': 'full_c32' if graph_c32 else 'batch1_requires_source_specific_validation' if allow_decode_graph else 'disabled',
            'schedule_mode': 'measured_dynamic' if profile is not None else 'explicit_fixed',
            'hot_cache':hot_cache,
            'decode_implementation':decode_implementation,
            'staged_max_batch':32 if runtime.staged_decode is not None else None,
            'staged_additional_slot_collective':runtime.staged_decode is not None,
            'staged_resources':dict(runtime.staged_decode.resources) if runtime.staged_decode is not None else None,
            **_hopper_metadata(runtime.hopper_decode),
            **_prefill_metadata(runtime),
            'tile_rows': 16 if kernel=='tiled' else 1,
            'tile_columns': 64 if kernel=='tiled' else None}


def _hopper_metadata(hopper):
    if hopper is None:
        return {}
    return {'candidate_source_sha256': hopper.candidate_source_sha256,
            'assembly_source_sha256': hopper.assembly_source_sha256,
            'numerical_policy': 'record_only',
            'hopper_max_batch': 32,
            'hopper_slot_collective': 'torch.distributed.all_reduce/SUM/tp_device_group',
            'hopper_resources': dict(hopper.resources)}


def _prefill_metadata(runtime):
    prefill = getattr(runtime, 'hopper_prefill', None)
    return {'prefill_implementation': getattr(runtime, 'prefill_implementation', 'persistent'),
            'prefill_source_sha256': prefill.candidate_source_sha256 if prefill is not None else None,
            'prefill_runtime_source_sha256': prefill.runtime_source_sha256 if prefill is not None else None,
            'prefill_resources': dict(prefill.resources) if prefill is not None else None,
            'additional_prefill_slot_collectives': prefill.additional_slot_collectives if prefill is not None else 0}


def execution_profile(model):
    runtime=model.weave_runtime
    profile=model.model.layers[0].mlp.experts.profile
    prop=runtime.torch.cuda.get_device_properties(runtime.device)
    return runtime.execution_profile_snapshot({'rank':runtime.rank,'kernel':runtime.kernel,
        'source_sha256':runtime.source_sha256,'graph_enabled':runtime.allow_decode_graph,
        'profile_bank_sha256':getattr(profile,'sha256',None),
        'schedule_mode':'measured_dynamic' if profile is not None else 'explicit_fixed',
        'fixed_plan':None if profile is not None else list(runtime.fixed_plan),
        'checkpoint_content_sha256':os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256'),
        'image_id':os.environ.get('WEAVE_IMAGE_ID'),'host_gpu_allocation':os.environ.get('WEAVE_HOST_ALLOCATION'),
        'device_key':f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}'})


def coverage(model):
    runtime=model.weave_runtime
    batch_counts=runtime.model_batch_counters.snapshot()
    staged=runtime.staged_decode
    hopper=getattr(runtime,'hopper_decode',None)
    slot_runtime=hopper if hopper is not None else staged
    result={'rank': model.weave_runtime.rank, 'kernel': model.weave_runtime.kernel,
            'source_sha256': model.weave_runtime.source_sha256,
            'graph_enabled': model.weave_runtime.allow_decode_graph,
            'schedule_mode': 'measured_dynamic' if model.model.layers[0].mlp.experts.profile is not None else 'explicit_fixed',
            'fixed_plan': None if model.model.layers[0].mlp.experts.profile is not None else list(model.weave_runtime.fixed_plan),
            'profile_checkpoint_content_sha256': os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256'),
            'profile_bank_sha256': getattr(model.model.layers[0].mlp.experts.profile,'sha256',None),
            'execution_profile_configuration':asdict(model.weave_runtime.trace_config) if model.weave_runtime.trace_config else None,
            'hot_cache':model.weave_runtime.hot_cache,
            'decode_implementation':runtime.decode_implementation,
            'staged_max_batch':32 if staged is not None else None,
            'staged_slot_collective':'torch.distributed.all_reduce/SUM/tp_device_group' if staged is not None else None,
            'additional_slot_collectives':slot_runtime.additional_slot_collectives if slot_runtime is not None else 0,
            'staged_resources':dict(staged.resources) if staged is not None else None,
            **_hopper_metadata(hopper),
            **_prefill_metadata(runtime),
            'model_batch_counts':batch_counts,
            'layers': {str(layer.mlp.layer_id): {
                'device_calls': layer.mlp.experts.device_calls.cpu().tolist(),
                'capture_calls': layer.mlp.experts.capture_calls,
                'python_decode_calls': layer.mlp.experts.coverage['decode'],
                'python_calls': dict(layer.mlp.experts.coverage),
                'implementation_calls':dict(layer.mlp.experts.implementation_calls)} for layer in model.model.layers}}
    if getattr(runtime, 'graph_c32', False):
        result.update(graph_mode='full', numerical_policy='record_only',
            graph_runtime_source_sha256=hopper.graph_runtime_source_sha256,
            device_slot_collectives=hopper.device_slot_collectives_snapshot())
    if runtime.model_batch_counters.snapshot()['sequence'] != batch_counts['sequence']:
        raise ValueError('service became active during model coverage snapshot')
    return result
