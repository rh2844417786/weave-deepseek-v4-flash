"""Opt-in native Marlin budgeting, installed before SGLang graph capture.

Graph replay never invokes these Python wrappers. The captured Marlin kernels
read current grouping and choose their virtual worker count on the GPU.
"""
from contextvars import ContextVar
from functools import wraps
import importlib
import os
from pathlib import Path

from .calibration import atomic_json
from .native_budget import METHOD, ROLES, STATE_SIZE, native_budget_source_sha256, state_row
from .native_budget_patch import load_budget_op, verify_sources

DECODE = ContextVar('weave_budget_decode', default=False)
LAYER = ContextVar('weave_budget_layer', default=None)


def budget_options(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get('WEAVE_NATIVE_BUDGET', 'off')
    if mode == 'off':
        if any(k in env for k in ('WEAVE_NATIVE_BUDGET_DIR', 'WEAVE_NATIVE_BUDGET_TARGET_PERCENT')):
            raise ValueError('native budget settings require adaptive/full mode')
        return None
    if (mode not in ('adaptive', 'full') or env.get('WEAVE_GRAPH_C32') != '1'
            or env.get('WEAVE_REQUEST_TIMING') != '1' or env.get('SGLANG_EXTERNAL_MODEL_PACKAGE')
            or any(env.get(k, '0') != '0' for k in ('WEAVE_NATIVE_P0', 'WEAVE_GRAPH_STAGE_PROFILE',
                   'WEAVE_FIXED_PREFIX', 'WEAVE_EXECUTION_PROFILE', 'WEAVE_FORWARD_PROFILE', 'WEAVE_STALL_DIAGNOSTIC'))
            or env.get('WEAVE_PROFILE') or env.get('WEAVE_REPLAY_CAPTURE_DIR')):
        raise ValueError('native budget requires native full graph without diagnostic/profile intervention')
    target = int(env.get('WEAVE_NATIVE_BUDGET_TARGET_PERCENT', '75'))
    directory = env.get('WEAVE_NATIVE_BUDGET_DIR', '')
    if not 1 <= target <= 100 or not Path(directory).is_absolute() or '..' in Path(directory).parts:
        raise ValueError('native budget needs target 1..100 and an absolute evidence directory')
    return dict(mode=mode, target_percent=target, directory=directory)


def install_dispatch(model, marlin_module, candidate, runtime, *, layers=43):
    blocks = model.model.layers
    if len(blocks) != layers:
        raise RuntimeError('native budget requires complete 43-layer model coverage')
    original = marlin_module.moe_wna16_marlin_gemm
    if getattr(original, '_weave_budget_dispatch', False):
        raise RuntimeError('native budget dispatch installed twice')

    @wraps(original)
    def dispatch(*args, **kwargs):
        state = LAYER.get()
        if state is None or state['runtime'] is not runtime:
            return original(*args, **kwargs)
        index = state['calls']
        if index >= 2:
            raise RuntimeError('native budget requires exactly two routed Marlin calls per layer')
        if (not kwargs.get('is_ep') or kwargs.get('mul_topk_weights') != bool(index)
                or (index == 1 and kwargs.get('top_k') != 1)):
            raise RuntimeError('native budget routed Marlin role/EP contract changed')
        state['calls'] += 1
        return candidate(*args, **kwargs, budget_state=runtime.states[state['layer'], ROLES[index]],
                         budget_target_percent=runtime.target_percent, budget_ep_size=4,
                         budget_mode=1 if runtime.mode == 'adaptive' else 2)

    dispatch._weave_budget_dispatch = True
    marlin_module.moe_wna16_marlin_gemm = dispatch
    for index, block in enumerate(blocks):
        original_mlp = block.mlp.forward

        @wraps(original_mlp)
        def mlp_forward(*args, _original=original_mlp, _layer=index, **kwargs):
            if not DECODE.get():
                return _original(*args, **kwargs)
            state = dict(runtime=runtime, layer=_layer, calls=0)
            token = LAYER.set(state)
            try:
                result = _original(*args, **kwargs)
                if state['calls'] != 2:
                    raise RuntimeError('native budget requires exactly two routed Marlin calls per layer')
                return result
            finally:
                LAYER.reset(token)
        block.mlp.forward = mlp_forward
    original_model = model.forward

    @wraps(original_model)
    def model_forward(*args, **kwargs):
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        if batch is None:
            raise RuntimeError('native budget requires explicit model forward mode')
        token = DECODE.set(bool(batch.forward_mode.is_decode()))
        try:
            return original_model(*args, **kwargs)
        finally:
            DECODE.reset(token)
    model.forward = model_forward


class NativeBudgetRuntime:
    def __init__(self, torch, rank, package_root, *, mode, target_percent, directory):
        self.torch, self.rank, self.package_root = torch, rank, Path(package_root)
        self.mode, self.target_percent = mode, target_percent
        self.source = native_budget_source_sha256()
        self.pins = verify_sources(package_root)
        self.device = torch.cuda.current_device()
        self.buffer = torch.zeros((43, 2, STATE_SIZE), dtype=torch.int64, device=self.device)
        self.states = {(i, role): self.buffer[i, j] for i in range(43) for j, role in enumerate(ROLES)}
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path(directory) / 'cache')))
        self.candidate = load_budget_op(package_root, cache / 'weave-native-budget' / f'rank{rank}')
        # The supported checkpoint has BF16 activations, EP and no expert bias.
        # Compile before capture; do not rely on a first captured call to compile.
        self.candidate.weave_prepare(torch.bfloat16, True, False)
        self.identity = dict(schema_version=1, method=METHOD, numerical_policy='record_only', rank=rank,
                             layer_count=43, source_sha256=self.source, upstream_sha256=self.pins,
                             overlay_sha256=self.candidate.weave_overlay_sha256,
                             policy=dict(mode=mode, target_percent=target_percent, ep_size=4, min_percent=25, buckets=8),
                             resource_scope='active_CTA_budget_not_physical_SM_partition',
                             policy_model='uncalibrated_live_route_share_heuristic',
                             paper_chunk_pipeline=False, paper_bubble_stealing=False)
        atomic_json(Path(directory) / f'identity-rank{rank}.json', self.identity)

    def verify(self):
        verify_sources(self.package_root)
        if native_budget_source_sha256() != self.source:
            raise RuntimeError('native budget implementation changed after compilation')

    def snapshot(self):
        # Caller holds GraphObserver.lock and rejects an active host forward.
        # This endpoint is used at idle measurement boundaries only.
        self.verify()
        with self.torch.cuda.device(self.device):
            self.torch.cuda.synchronize(self.device)
            rows = self.buffer.cpu().tolist()
        return dict(self.identity, rows=[state_row(i, role, rows[i][j])
                    for i in range(43) for j, role in enumerate(ROLES)])


def install_native_budget(model, torch, rank, package_root, **options):
    runtime = NativeBudgetRuntime(torch, rank, package_root, **options)
    module = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')
    install_dispatch(model, module, runtime.candidate, runtime)
    model.weave_native_budget = runtime
    return runtime
