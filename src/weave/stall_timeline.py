"""Bounded diagnostic timeline. CUDA is imported only by the real backend."""
from contextlib import contextmanager
import math
import threading
import time

class TorchBackend:
    def __init__(self, torch, device): self.torch, self.device = torch, device
    def event(self): return self.torch.cuda.Event(enable_timing=True)
    def synchronize(self): self.torch.cuda.synchronize(self.device)
    def capturing(self): return self.torch.cuda.is_current_stream_capturing()

class StallTimeline:
    def __init__(self, backend, *, rank, device, source_sha256, identity,
                 native_source_sha256='', max_calls=16000, max_samples=256):
        self.backend, self.rank, self.device = backend, rank, device
        self.source_sha256, self.identity = source_sha256, identity
        self.native_source_sha256 = native_source_sha256
        self.max_calls, self.max_samples = max_calls, max_samples
        self.lock = threading.RLock()
        self.events = [[backend.event() for _ in range(88)] for _ in range(max_samples)]
        self.anchor_event = backend.event()
        self.armed = False
        self.active = False
        self.current = None
        self.calls, self.samples, self.failures = [], [], []
        self.overflow = {'calls':False,'samples':False}
        self.anchor = None
        self.decode_count = 0
        self.current_phase = None
        self.current_layers = []
        self.coverage = {phase: [0]*43 for phase in ('prefill','decode')}
    def status(self):
        with self.lock: return {'armed':self.armed,'active':self.active,'rank':self.rank}
    def arm(self):
        with self.lock:
            if self.active or self.armed: raise RuntimeError('collector must be idle and disarmed')
            if self.backend.capturing(): raise RuntimeError('CUDA capture is unsupported')
            # Materialize PyTorch lazy CUDA event handles while idle, before anchor.
            for events in self.events:
                for event in events: event.record()
            self.backend.synchronize()
            self.calls, self.samples, self.failures = [], [], []
            self.overflow = {'calls':False,'samples':False}
            self.decode_count = 0
            self.coverage = {phase: [0]*43 for phase in ('prefill','decode')}
            before = time.perf_counter_ns()
            self.anchor_event.record()
            self.backend.synchronize()
            self.anchor = {'host_before_ns':before,'host_after_ns':time.perf_counter_ns()}
            self.armed = True
            return dict(status='armed',armed=True,rank=self.rank,source_sha256=self.source_sha256)
    def fail(self, reason):
        if reason not in self.failures: self.failures.append(reason)
    @contextmanager
    def forward(self, phase, rows):
        with self.lock:
            if self.active: raise RuntimeError('overlapping model invocation')
            self.active = True
            recording = self.armed
            self.current_phase = phase if recording else None
            self.current_layers = []
            call = None
            if recording:
                if self.backend.capturing():
                    self.active=False
                    self.fail('cuda_capture_detected')
                    raise RuntimeError('stall diagnostic requires eager execution')
                sequence = len(self.calls)
                sampled = phase == 'prefill' or self.decode_count % 32 == 0
                if phase == 'decode': self.decode_count += 1
                if sequence >= self.max_calls:
                    self.overflow['calls']=True; self.fail('call_capacity_exceeded')
                else:
                    call=dict(sequence=sequence,phase=phase,rows=int(rows),host_start_ns=time.perf_counter_ns(),host_end_ns=None,sample_index=None)
                    self.calls.append(call)
                if sampled:
                    if len(self.samples)>=self.max_samples:
                        self.overflow['samples']=True; self.fail('sample_capacity_exceeded')
                    elif call is not None:
                        idx=len(self.samples)
                        self.current=dict(sequence=sequence,phase=phase,rows=int(rows),index=idx,layers=[])
                        self.samples.append(self.current)
                        call['sample_index']=idx
                        self.events[idx][0].record()
        try:
            yield
        except BaseException:
            with self.lock:
                if recording: self.fail('model_forward_failed')
            raise
        finally:
            with self.lock:
                if self.current is not None:
                    self.events[self.current['index']][1].record()
                    if self.current['layers'] != list(range(43)): self.fail('incomplete_or_out_of_order_routed_layers')
                if recording and self.current_layers != list(range(43)):
                    self.fail('incomplete_or_out_of_order_routed_layers')
                self.current=None
                self.current_phase=None
                if call is not None: call['host_end_ns']=time.perf_counter_ns()
                self.active=False
    @contextmanager
    def routed(self, layer):
        if self.current_phase is not None:
            self.coverage[self.current_phase][layer] += 1
            self.current_layers.append(layer)
        sample=self.current
        if sample is not None:
            if layer in sample['layers']: self.fail('duplicate_routed_layer')
            sample['layers'].append(layer)
            self.events[sample['index']][2+2*layer].record()
        try: yield
        finally:
            if sample is not None: self.events[sample['index']][3+2*layer].record()
    def snapshot(self):
        with self.lock:
            if self.active: raise RuntimeError('snapshot requires idle model')
            if self.anchor is None: raise RuntimeError('snapshot requires a prior arm')
            if self.backend.capturing(): raise RuntimeError('snapshot forbidden during capture')
            self.armed=False
            self.backend.synchronize()
            samples=[]
            for sample in self.samples:
                events=self.events[sample['index']]
                def read(i, start=None):
                    value=(self.anchor_event if start is None else events[start]).elapsed_time(events[i])
                    if not math.isfinite(value) or value < 0:
                        self.fail('invalid_cuda_event_time')
                        return None
                    return value
                samples.append(dict(sequence=sample['sequence'],phase=sample['phase'],rows=sample['rows'],
                    gpu_start_ms=read(0),gpu_end_ms=read(1),gpu_duration_ms=read(1,0),routed=[dict(layer=l,start_ms=read(2+2*l),end_ms=read(3+2*l),duration_ms=read(3+2*l,2+2*l)) for l in sample['layers']]))
            return dict(schema_version=1,rank=self.rank,device=self.device,source_sha256=self.source_sha256,
                identity=self.identity,native_source_sha256=self.native_source_sha256,source_verified=True,
                clock='CLOCK_MONOTONIC',anchor=self.anchor,calls=list(self.calls),samples=samples,
                coverage={p:list(v) for p,v in self.coverage.items()},failures=list(self.failures),overflow=dict(self.overflow),expected_layers=43,
                numerical_policy='record_only',instrumented_diagnostic=True,used_for_formal_performance=False)
