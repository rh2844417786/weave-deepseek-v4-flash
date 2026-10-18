"""Eager staged decode orchestration and CPU-only serving coverage counters.

The workspace owns the online grouping, routed-product rounding and final
origin assembly. Its slot SUM is an additional collective before SGLang's
existing shared-expert/scaling/TP reduction, and is part of serving time.
"""
from collections import Counter
from threading import Lock


def validate_decode_configuration(*, decode_implementation, kernel, hot_cache,
                                  allow_decode_graph, fixed_plan, profile_path=None,
                                  execution_profile=None, replay_capture=False,
                                  forward_profile=False):
    if decode_implementation not in ('persistent', 'staged_direct', 'hopper_fp8'):
        raise ValueError('decode implementation must be persistent, staged_direct or hopper_fp8')
    if decode_implementation in ('staged_direct', 'hopper_fp8') and (
            kernel != 'tiled' or not hot_cache or allow_decode_graph
            or tuple(fixed_plan) != (1, 1, 0) or profile_path is not None
            or execution_profile is not None or replay_capture or forward_profile):
        raise ValueError(f'{decode_implementation} requires tiled, hot-cache, eager, fixed(1,1,0), '
                         'and no profile bank, execution profiling or replay capture')


def validate_prefill_configuration(*, prefill_implementation='persistent',
                                   decode_implementation, kernel, hot_cache,
                                   allow_decode_graph, fixed_plan, profile_path=None,
                                   execution_profile=None, replay_capture=False,
                                   forward_profile=False):
    if prefill_implementation not in ('persistent', 'hopper_fp8'):
        raise ValueError('prefill implementation must be persistent or hopper_fp8')
    if prefill_implementation == 'hopper_fp8':
        if decode_implementation != 'hopper_fp8':
            raise ValueError('hopper_fp8 prefill requires hopper_fp8 decode')
        validate_decode_configuration(decode_implementation=decode_implementation,
            kernel=kernel, hot_cache=hot_cache, allow_decode_graph=allow_decode_graph,
            fixed_plan=fixed_plan, profile_path=profile_path,
            execution_profile=execution_profile, replay_capture=replay_capture,
            forward_profile=forward_profile)


class StagedDecodeRuntime:
    """One workspace shared by sequential layers; bound handles own raw weights.

    Output storage is reused on the next layer invocation, after the current
    layer's outer SGLang operations have been queued on the same CUDA stream.
    No CPU tensor values, synchronization, or per-call output views are used.
    """
    max_batch = 32

    def __init__(self, peer_runtime, device_group):
        from .ops import extension
        self.torch = peer_runtime.torch
        self.device_group = device_group
        self.ws = extension().StagedWorkspace(peer_runtime.rank, peer_runtime.ep,
            self.max_batch, peer_runtime.h, peer_runtime.i, 256, peer_runtime.topk)
        self.resources = {key: value for key, value in self.ws.resources().items()
                          if key.startswith('online_')}
        slots, output = self.ws.online_slot_outputs, self.ws.online_output
        self._slots = (None,) + tuple(slots[:n * peer_runtime.topk]
                                     for n in range(1, self.max_batch + 1))
        self._outputs = (None,) + tuple(output[:n] for n in range(1, self.max_batch + 1))
        self.additional_slot_collectives = 0

    def bind_weights(self, raw_weights):
        return self.ws.bind_online_weights(list(raw_weights))

    def run(self, x, ids, routing, bound, clamp=10):
        batch = x.shape[0]
        if not 1 <= batch <= self.max_batch:
            raise ValueError('staged_direct decode supports batches 1..32 only')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('staged_direct requires eager decode')
        self.ws.run_online(x, ids, routing, bound, clamp)
        dist = self.torch.distributed
        dist.all_reduce(self._slots[batch], op=dist.ReduceOp.SUM, group=self.device_group)
        self.ws.finalize_online(batch)
        self.additional_slot_collectives += 1
        return self._outputs[batch]


class ModelBatchCounters:
    """Shape metadata only; snapshots are cumulative and require idle service."""
    def __init__(self):
        self._lock = Lock()
        self._current = None
        self._sequence = self._successful = self._failed = 0
        self._by_phase = {phase: Counter() for phase in ('prefill', 'decode', 'other')}
        self._decode_valid = Counter()

    def begin(self, phase, rows, valid_rows):
        with self._lock:
            if self._current is not None:
                raise ValueError('concurrent model forwards cannot share the staged workspace')
            if phase not in self._by_phase or type(rows) is not int or rows < 0:
                raise ValueError('model batch counters require phase and nonnegative row count')
            if type(valid_rows) is not int or not 0 <= valid_rows <= rows:
                raise ValueError('model valid rows must be an integer within the input rows')
            self._sequence += 1
            self._current = (phase, rows, valid_rows)

    def end(self, *, success):
        with self._lock:
            if self._current is None:
                raise ValueError('model forward coverage has no active forward')
            phase, rows, valid_rows = self._current
            if success:
                self._successful += 1
                self._by_phase[phase][str(rows)] += 1
                if phase == 'decode':
                    self._decode_valid[str(valid_rows)] += 1
            else:
                self._failed += 1
            self._current = None

    def snapshot(self):
        with self._lock:
            if self._current is not None:
                raise ValueError('service is active during model coverage snapshot')
            return {'active': False, 'sequence': self._sequence,
                    'successful_forwards': self._successful, 'failed_forwards': self._failed,
                    'by_phase': {phase: dict(counts) for phase, counts in self._by_phase.items()},
                    'decode_valid_batch_histogram': dict(self._decode_valid)}
