"""Opt-in native graph diagnostics; event timestamps and aggregate work only.

Captured events are reused by CUDA Graph. Each selected replay must be finished
and read before another replay can overwrite them. The graph observer provides
that serialization and its idle lock protects snapshot().
"""
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SPANS = ('moe', 'routed', 'shared', 'gate_up', 'activation', 'down',
         'routed_finalize', 'shared_merge', 'all_reduce', 'router', 'topk', 'grouping')
TIMING_ROUNDING_MS = 0.001
_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_FILES = ('src/weave/native_p0.py', 'src/weave/native_p0_hooks.py',
                  'src/weave/native_p0_trace.py', 'configs/native_p0_sources.json')


def native_p0_source_manifest():
    return {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
            if (_ROOT / name).is_file() else 'missing' for name in _PROFILE_FILES}


def native_p0_source_sha256():
    return hashlib.sha256(json.dumps(native_p0_source_manifest(), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


_LOADED_PROFILE_SOURCE_MANIFEST = native_p0_source_manifest()
_LOADED_PROFILE_SOURCE_SHA256 = native_p0_source_sha256()


def derive_layer_metrics(intervals):
    """Derive simultaneous branch contribution on one GPU clock."""
    result = {}
    if 'moe' in intervals:
        result['moe_ms'] = intervals['moe']['duration_ms']
    if 'routed' not in intervals or 'shared' not in intervals:
        return result
    routed, shared = intervals['routed'], intervals['shared']
    done = max(routed['end_ms'], shared['end_ms'])
    result.update(
        branch_overlap_ms=max(0., min(routed['end_ms'], shared['end_ms']) -
                              max(routed['start_ms'], shared['start_ms'])),
        shared_tail_ms=max(0., shared['end_ms'] - max(shared['start_ms'], routed['end_ms'])),
        routed_tail_ms=max(0., routed['end_ms'] - max(routed['start_ms'], shared['end_ms'])),
        branches_done_ms=done)
    if 'moe' in intervals:
        result['moe_post_branches_ms'] = max(0., intervals['moe']['end_ms'] - done)
    return result


class NativeP0Profiler:
    def __init__(self, torch, rank, stride=128, capacity=64, layers=43, trace=None):
        for name, value in (('stride', stride), ('capacity', capacity), ('layers', layers)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(rank) is not int or rank < 0:
            raise ValueError('rank must be a nonnegative integer')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('allocate profiler before graph capture')
        self.torch, self.rank, self.kind = torch, rank, 'native_p0'
        self.stride, self.capacity, self.layers_expected = stride, capacity, layers
        self.profile_source_sha256 = _LOADED_PROFILE_SOURCE_SHA256
        self.profile_source_manifest = deepcopy(_LOADED_PROFILE_SOURCE_MANIFEST)
        self._layers, self._metadata, self._present, self._grouping = {}, {}, {}, {}
        self._samples = deque(maxlen=capacity)
        self._pending, self._last_sequence = None, 0
        self.b32_seen = self.sampled_total = self.aborted_samples = 0
        self.trace, self._trace_active, self._trace_attempted = trace, False, False
        self._launch_trace = None
        self._outer = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        for event in self._outer:
            event.record()

    def register_layer(self, layer, metadata):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('register events before graph capture')
        if type(layer) is not int or layer < 0:
            raise ValueError('layer must be a nonnegative integer')
        if layer in self._layers:
            self.set_metadata(layer, metadata)
            return
        if len(self._layers) >= self.layers_expected:
            raise ValueError('too many registered layers')
        events = {name: tuple(self.torch.cuda.Event(enable_timing=True, external=True)
                             for _ in range(2)) for name in SPANS}
        # CUDA events allocate lazily. Materialize every event outside capture.
        for pair in events.values():
            for event in pair:
                event.record()
        self._layers[layer], self._metadata[layer], self._present[layer] = events, {}, set()
        self.set_metadata(layer, metadata)

    def set_metadata(self, layer, updates):
        if layer not in self._metadata:
            raise ValueError('register layer before setting metadata')
        # Prevent accidentally retaining GPU tensors or nonfinite scalars.
        clean = json.loads(json.dumps(updates, allow_nan=False))
        self._metadata[layer].update(clean)

    @contextmanager
    def span(self, layer, name):
        if name not in SPANS:
            raise ValueError(f'unknown native span: {name}')
        if not self.torch.cuda.is_current_stream_capturing():
            yield
            return
        if layer not in self._layers:
            raise RuntimeError('register events before graph capture')
        start, end = self._layers[layer][name]
        start.record()  # Implicit current stream is essential for shared branch.
        try:
            yield
        except BaseException:
            raise
        else:
            end.record()
            self._present[layer].add(name)

    def bind_grouping(self, layer, sorted_token_ids, expert_ids, num_tokens_post_pad,
                      block_size, valid_pair_capacity):
        if not self.torch.cuda.is_current_stream_capturing():
            return
        if layer not in self._layers:
            raise RuntimeError('register grouping layer before graph capture')
        if type(block_size) is not int or block_size <= 0:
            raise ValueError('grouping block_size must be positive')
        if type(valid_pair_capacity) is not int or valid_pair_capacity < 0:
            raise ValueError('valid_pair_capacity must be nonnegative')
        self._grouping[layer] = (sorted_token_ids, expert_ids, num_tokens_post_pad,
                                 block_size, valid_pair_capacity)

    def begin_replay(self, sequence, batch):
        if self._pending is not None:
            raise RuntimeError('previous sampled replay has not finished')
        if type(sequence) is not int or sequence <= self._last_sequence:
            raise ValueError('actual replay sequence must increase')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('replay sampling is forbidden during capture')
        self._last_sequence = sequence
        if batch != 32:
            return False
        self.b32_seen += 1
        if self.b32_seen % self.stride:
            return False
        if len(self._layers) != self.layers_expected:
            raise RuntimeError('sample requires all expected layers registered')
        if any('moe' not in names for names in self._present.values()):
            raise RuntimeError('sample requires captured moe span for every layer')
        self._pending = (sequence, batch, self.b32_seen)
        if self.trace is not None and not self._trace_attempted:
            self._trace_attempted = True
            try:
                self.trace.begin()
                self._trace_active = True
            except Exception as exc:
                self._launch_trace = dict(status='failed', error_type=type(exc).__name__)
                self.abort_replay(sequence, batch)
                raise
        self._outer[0].record()
        return True

    def abort_replay(self, sequence, batch):
        if self._pending is None:
            return
        if self._pending[:2] != (sequence, batch):
            raise RuntimeError('aborted replay does not match pending sample')
        self._pending = None
        self.aborted_samples += 1
        if self._trace_active:
            try:
                self.trace.abort()
            finally:
                self._trace_active = False
                self._launch_trace = dict(status='aborted')

    @staticmethod
    def _elapsed(start, end):
        value = float(start.elapsed_time(end))
        if not math.isfinite(value):
            raise RuntimeError('nonfinite CUDA event timing')
        return value

    @staticmethod
    def _host_values(tensor):
        return tensor.detach().cpu().reshape(-1).tolist()

    def _grouping_summary(self, layer):
        sorted_ids, experts, padded, block, capacity = self._grouping[layer]
        sorted_ids = self._host_values(sorted_ids)
        experts = self._host_values(experts)
        padded = self._host_values(padded)
        if len(padded) != 1 or int(padded[0]) != padded[0]:
            raise RuntimeError('malformed native padded count')
        padded = int(padded[0])
        if padded < 0 or padded % block or padded > len(sorted_ids) or padded // block > len(experts):
            raise RuntimeError('malformed native grouping capacity')
        rows, valid_groups = Counter(), 0
        for index in range(padded // block):
            expert = int(experts[index])
            if expert == -1:
                continue
            if expert < 0:
                raise RuntimeError('malformed native expert index')
            valid = sum(0 <= token < capacity for token in sorted_ids[index*block:(index+1)*block])
            valid_groups += 1
            if valid:
                rows[expert] += valid
        histogram = Counter(rows.values())
        return dict(valid_local_pairs=sum(rows.values()), valid_groups=valid_groups,
                    active_experts=len(rows), max_rows=max(rows.values(), default=0),
                    row_histogram={str(key): value for key, value in sorted(histogram.items())},
                    num_tokens_post_pad=padded, block_size=block, valid_pair_capacity=capacity)

    def finish_replay(self, sequence, batch):
        if self._pending is None:
            return None
        if self._pending[:2] != (sequence, batch):
            raise RuntimeError('finished replay does not match pending sample')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('replay sampling is forbidden during capture')
        launch_trace_active = self._trace_active
        self._outer[1].record()
        self._outer[1].synchronize()
        whole = self._elapsed(*self._outer)
        if whole < 0:
            raise RuntimeError('negative whole graph CUDA event timing')
        rows, previous_end = {}, 0.
        for layer, events in sorted(self._layers.items()):
            intervals = {}
            for name in SPANS:
                if name not in self._present[layer]:
                    continue
                start, end = events[name]
                lo, hi = (self._elapsed(self._outer[0], event) for event in (start, end))
                duration = self._elapsed(start, end)
                if (lo < -TIMING_ROUNDING_MS or hi > whole + TIMING_ROUNDING_MS or
                        duration < 0 or hi < lo or abs(hi-lo-duration) > TIMING_ROUNDING_MS):
                    raise RuntimeError('stale or invalid native CUDA event window')
                intervals[name] = dict(start_ms=lo, end_ms=hi, duration_ms=duration)
            moe = intervals['moe']
            if moe['start_ms'] < previous_end - TIMING_ROUNDING_MS:
                raise RuntimeError('native layer event order differs')
            previous_end = moe['end_ms']
            for name, item in intervals.items():
                if (item['start_ms'] < moe['start_ms'] - TIMING_ROUNDING_MS or
                        item['end_ms'] > moe['end_ms'] + TIMING_ROUNDING_MS):
                    raise RuntimeError(f'native {name} event outside moe window')
            rows[str(layer)] = dict(intervals=intervals, **derive_layer_metrics(intervals))
        # Freeze trace before D2H grouping copies add irrelevant launch records.
        if self._trace_active:
            try:
                self._launch_trace = self.trace.finish()
            except Exception as exc:
                self._launch_trace = dict(status='failed', error_type=type(exc).__name__)
                raise
            finally:
                self._trace_active = False
        for layer in self._grouping:
            rows[str(layer)]['grouping'] = self._grouping_summary(layer)
        sample = dict(sequence=sequence, batch=32, batch_size=32, b32_ordinal=self._pending[2],
                      whole_graph_ms=whole, layers=rows, launch_trace_active=launch_trace_active)
        self._samples.append(sample)
        self.sampled_total += 1
        self._pending = None
        return deepcopy(sample)

    def snapshot(self):
        if self._pending is not None:
            raise RuntimeError('native snapshot requires an idle replay boundary')
        metadata = {str(layer): dict(deepcopy(self._metadata[layer]),
                                    captured_spans=sorted(self._present[layer]))
                    for layer in self._layers}
        return dict(schema_version=1, kind=self.kind, rank=self.rank,
                    measurement_mode='instrumented_diagnostic', numerical_policy='record_only',
                    formal_performance_claim=False, external_events=True,
                    profile_source_sha256=self.profile_source_sha256,
                    profile_source_manifest=deepcopy(self.profile_source_manifest),
                    configuration=dict(stride=self.stride, capacity=self.capacity,
                                       layers=self.layers_expected, captured_batch_size=32),
                    b32_seen=self.b32_seen, sampled_total=self.sampled_total,
                    aborted_samples=self.aborted_samples, layer_metadata=metadata,
                    launch_trace=deepcopy(self._launch_trace),
                    samples=deepcopy(list(self._samples)))
