"""CPU-only full decode graph accounting; never reads device tensor values."""
from collections import Counter
from contextlib import contextmanager
from threading import RLock
import os

BATCHES = list(range(1, 33))


def graph_stage_options(environ=None):
    """Validate the explicit diagnostic environment without importing Torch."""
    env = os.environ if environ is None else environ
    enabled = env.get('WEAVE_GRAPH_STAGE_PROFILE', '0')
    if enabled not in ('0', '1'):
        raise ValueError('WEAVE_GRAPH_STAGE_PROFILE must be 0 or 1')
    if enabled == '0':
        if any(name in env for name in ('WEAVE_GRAPH_STAGE_STRIDE', 'WEAVE_GRAPH_STAGE_CAPACITY')):
            raise ValueError('graph stage stride/capacity require WEAVE_GRAPH_STAGE_PROFILE=1')
        return None
    if env.get('WEAVE_GRAPH_C32') != '1' or env.get('WEAVE_REQUEST_TIMING') != '1':
        raise ValueError('graph stage profiling requires graph C32 and request timing')
    if (any(env.get(name) == '1' for name in
            ('WEAVE_EXECUTION_PROFILE', 'WEAVE_FORWARD_PROFILE', 'WEAVE_STALL_DIAGNOSTIC'))
            or env.get('WEAVE_PROFILE') or env.get('WEAVE_REPLAY_CAPTURE_DIR')):
        raise ValueError('graph stage profiling cannot mix profiles, capture or other diagnostics')
    options = {}
    for key, default, maximum in (('stride', 128, 65536), ('capacity', 64, 256)):
        try:
            value = int(env.get('WEAVE_GRAPH_STAGE_' + key.upper(), str(default)))
        except (TypeError, ValueError) as error:
            raise ValueError('graph stage ' + key + ' must be an integer') from error
        if not 1 <= value <= maximum:
            raise ValueError(f'graph stage {key} must be in 1..{maximum}')
        options[key] = value
    return options

def canonical_graph_config():
    return {'decode': {'backend': 'full', 'max_bs': 32, 'bs': BATCHES.copy()},
            'prefill': {'backend': 'disabled'}}


def validate_graph_configuration(args, parallel=None):
    config = args.cuda_graph_config
    if (config.decode.backend != 'full' or config.decode.max_bs != 32
            or list(config.decode.bs) != BATCHES or config.prefill.backend != 'disabled'
            or args.max_running_requests != 32 or not args.disable_cuda_graph_padding):
        raise ValueError('graph C32 requires full exact batches 1..32, no padding, disabled prefill graph and max running 32')
    for name in ('enable_torch_compile', 'enable_dp_attention', 'enable_dp_lm_head',
                 'enable_lora', 'enable_two_batch_overlap', 'enable_single_batch_overlap',
                 'enable_pdmux', 'enable_profile_cuda_graph', 'enable_eplb'):
        if getattr(args, name, False):
            raise ValueError('graph C32 unsupported option: ' + name)
    if getattr(args, 'speculative_algorithm', None) or getattr(args, 'pp_size', 1) != 1:
        raise ValueError('graph C32 requires no speculative decoding or pipeline parallelism')
    if parallel is not None and any(getattr(parallel, k, None) != v for k, v in
            {'tp_size': 4, 'moe_ep_size': 4, 'moe_tp_size': 1, 'attn_dp_size': 1}.items()):
        raise ValueError('graph C32 requires TP4 EP4 expert TP1 attention DP1')


class GraphObserver:
    def __init__(self, rank, sources, observer_source, stage_profiler=None):
        self.rank, self.sources, self.observer_source = rank, sources, observer_source
        self.lock = RLock()
        self.active = self.sequence = 0
        self.capturing = False
        self.captured = set()
        self.counts = Counter()
        self.raw, self.buckets, self.forwards = Counter(), Counter(), Counter()
        self.stage_profiler = stage_profiler

    @contextmanager
    def operation(self):
        with self.lock:
            self.active += 1
            self.sequence += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
                self.sequence += 1

    def snapshot(self):
        with self.lock:
            if self.active:
                raise ValueError('graph service is active during coverage snapshot')
            result = {'rank': self.rank, 'graph_mode': 'full', 'backend': 'FullCudaGraphBackend',
                    'numerical_policy': 'record_only', 'source_sha256': dict(self.sources),
                    'observer_source_sha256': self.observer_source, 'active': False,
                    'sequence': self.sequence, 'captured_batches': sorted(self.captured),
                    **{k: self.counts[k] for k in ('capture_attempts', 'capture_successes',
                       'capture_failures', 'successful_replays', 'failed_replays')},
                    'raw_batch_histogram': dict(self.raw), 'captured_batch_histogram': dict(self.buckets),
                    'model_forwards': {k: self.forwards[k] for k in
                        ('capture', 'warmup', 'eager_decode', 'prefill', 'other')}}
            if self.stage_profiler is not None:
                result['stage_profile'] = self.stage_profiler.snapshot()
            return result
