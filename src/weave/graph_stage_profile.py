"""Opt-in external CUDA-event diagnostics for exact B32 graph replay.

No tensor contents leave this module. Timing calls are made only after the sampled
whole graph has finished; captured layer events are overwritten on every replay.
The unchanged graph/eager/CUDA implementations retain their own source identities.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .hopper_graph import HopperGraphDecodeRuntime, graph_runtime_source_sha256
from .hopper_finalize import finalize_slots

# CUDA event timestamp precision is approximately 0.5 us. This tolerance covers
# timestamp rounding in timing closure only, never numerical output differences.
TIMING_ROUNDING_MS = 0.001


def graph_stage_source_sha256():
    files = {'graph_stage_profile.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'graph_runtime': graph_runtime_source_sha256()}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


_LOADED_PROFILE_SOURCE_SHA256 = graph_stage_source_sha256()


class GraphStageProfiler:
    def __init__(self, torch, rank, kind, stride=128, capacity=64, layers=43):
        if kind not in ('native', 'weave'):
            raise ValueError('stage profiler kind must be native or weave')
        for name, value in (('stride', stride), ('capacity', capacity), ('layers', layers)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(rank) is not int or rank < 0:
            raise ValueError('rank must be a nonnegative integer')
        self.torch, self.rank, self.kind = torch, rank, kind
        self.stride, self.capacity = stride, capacity
        self.layers_expected = layers if kind == 'weave' else 0
        self.profile_source_sha256 = _LOADED_PROFILE_SOURCE_SHA256
        self._layers, self._bounds = {}, {}
        self._samples = deque(maxlen=capacity)
        self.b32_seen = self.sampled_total = self.aborted_samples = 0
        self._pending = None
        self._outer = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        # torch events are lazy: record before any graph capture can begin.
        for event in self._outer:
            event.record()

    def register_layer(self, layer, bound):
        if self.kind != 'weave':
            raise ValueError('native stage profiler only measures the whole graph')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('register stage events before graph capture')
        if type(layer) is not int or layer < 0:
            raise ValueError('layer must be a nonnegative integer')
        if layer in self._layers:
            if self._layers[layer][0] is bound:
                return
            raise ValueError('layer already registered with a different bound handle')
        if id(bound) in self._bounds:
            raise ValueError('bound handle already registered to another layer')
        if len(self._layers) >= self.layers_expected:
            raise ValueError('too many registered layers')
        events = tuple(self.torch.cuda.Event(enable_timing=True, external=True) for _ in range(4))
        for event in events:
            event.record()
        self._layers[layer] = (bound, events)
        self._bounds[id(bound)] = layer

    def events_for(self, bound):
        layer = self._bounds.get(id(bound))
        if layer is None or self._layers[layer][0] is not bound:
            raise RuntimeError('B32 bound handle must register stage events before capture')
        return self._layers[layer][1]

    def begin_replay(self, sequence, batch):
        if self._pending is not None:
            raise RuntimeError('previous sampled replay has not finished')
        if type(sequence) is not int or sequence <= 0:
            raise ValueError('actual replay sequence must be a positive integer')
        if batch != 32:
            return False
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('replay sampling is forbidden during capture')
        self.b32_seen += 1
        if self.b32_seen % self.stride:
            return False
        if len(self._layers) != self.layers_expected:
            raise RuntimeError('sample requires all expected layers registered')
        self._pending = (sequence, batch, self.b32_seen)
        self._outer[0].record()
        return True

    def abort_replay(self, sequence, batch):
        if self._pending is not None:
            if self._pending[:2] != (sequence, batch):
                raise RuntimeError('aborted replay does not match pending sample')
            self._pending = None
            self.aborted_samples += 1

    @staticmethod
    def _duration(start, end):
        value = float(start.elapsed_time(end))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError('nonfinite or negative CUDA event duration')
        return value

    def finish_replay(self, sequence, batch):
        if self._pending is None:
            return None
        if self._pending[:2] != (sequence, batch):
            raise RuntimeError('finished replay does not match pending sample')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('replay sampling is forbidden during capture')
        ordinal = self._pending[2]
        self._outer[1].record()
        self._outer[1].synchronize()
        # No graph replay or eager runtime call may interleave this read block.
        whole = self._duration(*self._outer)
        rows = []
        previous_end = self._outer[0]
        for layer, (_, events) in sorted(self._layers.items()):
            # Reject stale or out-of-order captured timestamps, even when their
            # standalone durations happen to look plausible.
            self._duration(previous_end, events[0])
            self._duration(events[3], self._outer[1])
            previous_end = events[3]
            phases = [self._duration(events[index], events[index + 1]) for index in range(3)]
            total = self._duration(events[0], events[3])
            if abs(sum(phases) - total) > TIMING_ROUNDING_MS:
                raise RuntimeError('CUDA layer event timing closure failed')
            rows.append(dict(layer=layer, expert_pipeline_ms=phases[0], slot_sum_ms=phases[1],
                             finalize_ms=phases[2], routed_total_ms=total))
        routed = sum(row['routed_total_ms'] for row in rows)
        residual = whole - routed
        if residual < -TIMING_ROUNDING_MS:
            raise RuntimeError('routed stage timing exceeds whole graph window')
        sample = dict(sequence=sequence, b32_ordinal=ordinal, batch=batch, whole_graph_ms=whole,
                      layers=rows, routed_total_ms=routed, residual_ms=max(0., residual))
        self._samples.append(sample)
        self.sampled_total += 1
        self._pending = None
        return deepcopy(sample)

    def snapshot(self):
        if self._pending is not None:
            raise RuntimeError('stage snapshot requires an idle replay boundary')
        return dict(schema_version=1, measurement_mode='instrumented_diagnostic',
                    numerical_policy='record_only', profile_source_sha256=self.profile_source_sha256,
                    rank=self.rank, kind=self.kind, stride=self.stride, capacity=self.capacity,
                    layers_expected=self.layers_expected, registered_layers=sorted(self._layers),
                    external_events=True, timing_rounding_tolerance_ms=TIMING_ROUNDING_MS,
                    b32_seen=self.b32_seen, sampled_total=self.sampled_total,
                    aborted_samples=self.aborted_samples, samples=deepcopy(list(self._samples)))


class HopperStageDecodeRuntime(HopperGraphDecodeRuntime):
    """Separate diagnostic runtime; expert_pipeline includes the whole ws.run."""
    def __init__(self, peer_runtime, device_group, profiler, experts=256):
        super().__init__(peer_runtime, device_group, experts)
        if profiler.kind != 'weave' or profiler.rank != self.rank:
            raise ValueError('stage runtime requires the matching Weave rank profiler')
        self.profiler = profiler
        self.profile_source_sha256 = _LOADED_PROFILE_SOURCE_SHA256

    def register_layer(self, layer, bound):
        self.profiler.register_layer(layer, bound)

    def run(self, x, ids, routing, bound, clamp=10):
        batch = x.shape[0]
        if batch != 32:
            return super().run(x, ids, routing, bound, clamp)
        capturing = self.torch.cuda.is_current_stream_capturing()
        if capturing and batch not in self._eager_warmed_batches:
            raise RuntimeError('hopper graph decode batch requires successful eager warmup')
        events = self.profiler.events_for(bound)
        events[0].record()
        self.ws.run(x, ids, routing, bound, clamp)
        events[1].record()
        dist = self.torch.distributed
        dist.all_reduce(self._slots[batch], op=dist.ReduceOp.SUM, group=self.device_group)
        events[2].record()
        self.device_slot_collectives.add_(1)
        finalize_slots(self._slots[batch], self._output, batch, self.hidden, self.topk, self.rank, self.ep)
        events[3].record()
        if not capturing:
            self.additional_slot_collectives += 1
            self._eager_warmed_batches.add(batch)
        return self._outputs[batch]
