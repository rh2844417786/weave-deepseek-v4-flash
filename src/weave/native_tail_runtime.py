"""Native decode integration for native_marlin_peer_tail_v1.

Only bound MoE/model forwards are wrapped. There is no process-wide Marlin or
all-reduce replacement. Initial native autotune runs unchanged; tail warmup and
calibration start in capture_one after SGLang enables its distributed graph
context. Captured graphs replay the kernels without invoking Python.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock

from .calibration import atomic_json
from .native_tail_patch import METHOD, ROOT, verify_sources
from .native_tail_front import load_decode_front

DECODE = ContextVar('weave_native_tail_decode', default=None)
GENERATE = 'python3 scripts/run_native_tail.py --work-root /srv/weave-work --records <prepared/main.jsonl> --execute'


def tail_options(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get('WEAVE_NATIVE_TAIL', 'off')
    if mode == 'off':
        if any(k in env for k in ('WEAVE_NATIVE_TAIL_DIRECTORY', 'WEAVE_NATIVE_TAIL_REPEATS')):
            raise ValueError('native tail settings require static/calibration/measured mode')
        return None
    if (mode not in ('static', 'calibration', 'measured')
            or env.get('WEAVE_GRAPH_C32') != '1' or env.get('WEAVE_REQUEST_TIMING') != '1'
            or env.get('WEAVE_NATIVE_BUDGET', 'off') != 'off'
            or env.get('SGLANG_EXTERNAL_MODEL_PACKAGE')
            or any(env.get(k, '0') != '0' for k in ('WEAVE_NATIVE_P0', 'WEAVE_GRAPH_STAGE_PROFILE',
                'WEAVE_FIXED_PREFIX', 'WEAVE_EXECUTION_PROFILE', 'WEAVE_FORWARD_PROFILE', 'WEAVE_STALL_DIAGNOSTIC'))
            or env.get('WEAVE_PROFILE') or env.get('WEAVE_REPLAY_CAPTURE_DIR')):
        raise ValueError('native tail requires native full graph without stacked methods or diagnostics')
    directory = env.get('WEAVE_NATIVE_TAIL_DIRECTORY', '')
    if not directory:
        raise ValueError('native tail measured tables or explicit calibration directory required; generate with ' + GENERATE)
    if not Path(directory).is_absolute() or '..' in Path(directory).parts:
        raise ValueError('native tail directory must be absolute')
    repeats = int(env.get('WEAVE_NATIVE_TAIL_REPEATS', '7'))
    if repeats < 2:
        raise ValueError('native tail calibration needs at least two repeats')
    return dict(mode=mode, directory=directory, repeats=repeats)


def native_tail_source_sha256():
    paths = sorted((ROOT / 'src/weave').glob('native_tail*.py'))
    paths += sorted(p for p in (ROOT / 'experimental/native_tail').rglob('*') if p.is_file())
    paths += [ROOT / name for name in ('configs/native_tail_sources.json','src/weave/native_cost_runtime.py','src/weave/native_cost_calibration.py','src/weave/graph_hooks.py','src/weave/graph_observer.py','configs/runtime.json')]
    digest = hashlib.sha256()
    for path in paths:
        # Exclude interpreter/cache files; only delivered implementation inputs.
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def install_tail_dispatch(model, runtime, *, layers=43):
    if len(model.model.layers) != layers:
        raise RuntimeError('native tail requires complete model layer coverage')
    if getattr(model.forward, '_weave_native_tail_dispatch', False):
        raise RuntimeError('native tail already installed')
    if hasattr(model, 'weave_native_cost') or hasattr(model, 'weave_native_budget'):
        raise RuntimeError('native tail cannot stack another native budget runtime')
    for index, block in enumerate(model.model.layers):
        original = block.mlp.forward
        @wraps(original)
        def mlp_forward(*args, _original=original, _layer=index, **kwargs):
            state = DECODE.get()
            if state is None or state[0] is not runtime or not runtime.active():
                return _original(*args, **kwargs)
            batch = state[1]
            runtime.maybe_calibrate(_layer, batch, _original, args, kwargs)
            return runtime.invoke_tail(_layer, _original, args, kwargs)
        block.mlp.forward = mlp_forward
    original_model = model.forward
    @wraps(original_model)
    def model_forward(*args, **kwargs):
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        if batch is None:
            raise RuntimeError('native tail requires explicit model forward mode')
        state = (runtime, int(batch.batch_size)) if batch.forward_mode.is_decode() else None
        token = DECODE.set(state)
        try:
            return original_model(*args, **kwargs)
        finally:
            DECODE.reset(token)
    model_forward._weave_native_tail_dispatch = True
    model.forward = model_forward


class NativeTailRuntime:
    def __init__(self, model, torch, rank, package_root, *, mode, directory, repeats=7):
        from .native_cost_runtime import classify_layers
        from .native_tail_executor import NativeTailExecutor
        self.model, self.torch, self.rank = model, torch, rank
        self.package_root = Path(package_root)
        self.mode, self.directory, self.repeats = mode, Path(directory), repeats
        self.source = native_tail_source_sha256()
        self.pins = verify_sources(self.package_root)
        self.groups, self.layer_groups = classify_layers(model)
        self.front = load_decode_front(torch, self.package_root)
        for block in model.model.layers:
            self.front.validate_layer(block.mlp)
        distributed = importlib.import_module('sglang.srt.distributed')
        self.tp_group = distributed.get_tp_group()
        if (self.tp_group.world_size != 4 or self.tp_group.rank_in_group != rank
                or self.tp_group.cpu_group is None):
            raise RuntimeError('native tail requires the existing four-rank TP CPU bootstrap group')
        self.group = self.tp_group.cpu_group
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(self.directory / 'cache')))
        self.executor = NativeTailExecutor(torch, rank, self.package_root, self.group,
            max_batch=32, cache_root=cache / METHOD / f'rank{rank}', directory=self.directory,
            mode=mode, repeats=repeats)
        self.executor.configure_groups(self.groups,self.layer_groups)
        self.host_tail_calls = [0] * 43
        self.identity = dict(schema_version=1, method=METHOD, numerical_policy='record_only',
            rank=rank, layer_count=43, source_sha256=self.source, upstream_sha256=self.pins,
            mode=mode, groups=self.groups,
            scope='native_MoE_decode_tail_not_full_Weave_paper_scheduler',
            resource_scope='resident_worker_quotas_not_exact_physical_SM_partition',
            shared_contract='FP8_TP4_partial_joined_before_tail',
            scaling_contract='topk_weight_in_down_then_BF16_slot_sum_then_outer_routed_scale_and_shared',
            coverage_contract='one_native_gate_up_one_tail_per_decode_layer')
        evidence=Path(os.environ.get('WEAVE_NATIVE_TAIL_EVIDENCE_DIR',str(self.directory)))
        atomic_json(evidence / f'identity-rank{rank}.json', self.identity)
        self._init_cleanup(evidence)

    def active(self):
        observer = getattr(self.model, 'weave_graph_observer', None)
        return bool(observer is not None and (observer.capturing or observer.captured))

    def maybe_calibrate(self, layer, batch, original, args, kwargs):
        if self.mode != 'calibration':
            return
        observer = getattr(self.model, 'weave_graph_observer', None)
        if (observer is None or not observer.capturing
                or self.torch.cuda.is_current_stream_capturing()):
            return
        from .native_tail_calibration import calibrate_layer
        calibrate_layer(self, layer, batch, original, args, kwargs)

    def invoke_native(self, layer, original, args, kwargs):
        return original(*args, **kwargs)

    def invoke_tail(self, layer, original, args, kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        hidden_states = inputs.pop('hidden_states')
        mlp = self.model.model.layers[layer].mlp
        if mlp.layer_id != layer:
            raise RuntimeError('native tail layer identity mismatch')
        result = self.front(mlp, hidden_states, self.executor, **inputs)
        self.host_tail_calls[layer] += 1
        return result

    def verify(self):
        verify_sources(self.package_root)
        if native_tail_source_sha256() != self.source:
            raise RuntimeError('native tail implementation changed after initialization')

    def snapshot(self):
        # Called by the graph endpoint only at idle measurement boundaries.
        self.verify()
        result = self.executor.snapshot()
        return dict(self.identity, executor=result,
            host_warmup_capture_tail_calls=list(self.host_tail_calls))

    def _init_cleanup(self, evidence):
        self._cleanup_lock, self._cleanup_operation = Lock(), Lock()
        self._cleanup_path = Path(evidence) / f'cleanup-rank{self.rank}.json'
        self._cleanup_state = dict(schema_version=1, protocol='http_two_phase_v1',
            rank=self.rank, status='not_started', phase='not_started',
            graphs_owned=None, graphs_reset=0, peers_disconnected=False,
            ipc_closed=False, completed_phases=[])
        self._cleanup_update()

    def cleanup_status(self):
        # This lock is never held during graph reset, synchronization or IPC calls.
        with self._cleanup_lock:
            return deepcopy(self._cleanup_state)

    def _cleanup_update(self, **changes):
        with self._cleanup_lock:
            self._cleanup_state.update(changes)
            self._cleanup_state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(self._cleanup_path, self._cleanup_state)

    def _cleanup_step(self, phase, operation):
        self._cleanup_update(phase=phase)
        operation()
        completed = self.cleanup_status()['completed_phases']
        self._cleanup_update(completed_phases=completed + [phase])

    def prepare_close(self):
        """Destroy graphs and disconnect peers locally; do not free exported memory.

        A live SGLang scheduler still broadcasts on its TP CPU group while idle.
        Service teardown must therefore never enter that group's collectives.
        The controller supplies the all-rank barrier between prepare and commit.
        """
        with self._cleanup_operation:
            state = self.cleanup_status()
            if state['status'] in ('prepared', 'complete'):
                return state
            if state['status'] == 'failed':
                raise RuntimeError('native tail cleanup previously failed; stop container')
            try:
                self._cleanup_update(status='preparing', phase='graph_owner')
                backend = getattr(self, 'graph_backend', None)
                if backend is None:
                    raise RuntimeError('native tail graph owner missing during cleanup')
                graphs = list(backend._graphs.values())
                self._cleanup_update(graphs_owned=len(graphs))
                # HTTP threads may default to device 0; select this rank explicitly.
                with self.torch.cuda.device(self.rank):
                    self._cleanup_step('synchronize_before_reset', lambda: self.torch.cuda.synchronize(self.rank))
                    def reset_graphs():
                        for index, graph in enumerate(graphs):
                            graph.reset()
                            self._cleanup_update(graphs_reset=index + 1)
                    self._cleanup_step('reset_graphs', reset_graphs)
                    self._cleanup_step('synchronize_after_reset', lambda: self.torch.cuda.synchronize(self.rank))
                    self._cleanup_step('disconnect_peers', self.executor.disconnect_peers)
                    self._cleanup_update(peers_disconnected=True)
                    self._cleanup_step('clear_backend', backend.cleanup)
                self._cleanup_update(status='prepared', phase='prepared')
                return self.cleanup_status()
            except Exception as exc:
                self._cleanup_update(status='failed', error_type=type(exc).__name__)
                raise

    def close(self):
        """Commit local release only after the HTTP controller confirms four prepares."""
        with self._cleanup_operation:
            state = self.cleanup_status()
            if state['status'] == 'complete':
                return state
            if state['status'] != 'prepared':
                raise RuntimeError('native tail cleanup must prepare all ranks before commit')
            try:
                self._cleanup_update(status='committing')
                with self.torch.cuda.device(self.rank):
                    self._cleanup_step('release_local', self.executor.release_local)
                self._cleanup_update(status='complete', phase='complete', ipc_closed=True)
                return self.cleanup_status()
            except Exception as exc:
                self._cleanup_update(status='failed', error_type=type(exc).__name__)
                raise


def install_native_tail(model, torch, rank, package_root, **options):
    runtime = NativeTailRuntime(model, torch, rank, package_root, **options)
    install_tail_dispatch(model, runtime)
    model.weave_native_tail = runtime
    return runtime
