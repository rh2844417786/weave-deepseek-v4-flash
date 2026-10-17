import unittest
from src.weave.stall_timeline import StallTimeline

class Event:
    tick = 0
    def record(self):
        Event.tick += 1
        self.tick = Event.tick
    def elapsed_time(self, other): return other.tick - self.tick

class Backend:
    def __init__(self): self.syncs = 0
    def event(self): return Event()
    def synchronize(self): self.syncs += 1
    def capturing(self): return False

class TimelineTests(unittest.TestCase):
    def make(self, **kw):
        b = Backend()
        return b, StallTimeline(b, rank=0, device=0, source_sha256='test', identity={}, **kw)
    def call(self, c, phase='prefill'):
        with c.forward(phase, 2):
            for layer in range(43):
                with c.routed(layer): pass
    def test_sample_selection_and_event_order(self):
        b,c = self.make()
        c.arm()
        synced = b.syncs
        self.call(c)
        for _ in range(33): self.call(c, 'decode')
        self.assertEqual(b.syncs, synced)
        r = c.snapshot()
        self.assertEqual(len(r['calls']),34)
        self.assertEqual([s['sequence'] for s in r['samples']], [0,1,33])
        self.assertEqual(len(r['samples'][0]['routed']),43)
        self.assertLess(r['samples'][0]['gpu_start_ms'],r['samples'][0]['routed'][0]['start_ms'])
        self.assertEqual(r['failures'],[])
    def test_direct_duration_preserves_small_span_after_large_anchor(self):
        import struct
        class FloatEvent(Event):
            def elapsed_time(self, other):
                return struct.unpack('f', struct.pack('f', other.tick-self.tick))[0]
        class FloatBackend(Backend):
            def event(self): return FloatEvent()
        c=StallTimeline(FloatBackend(),rank=0,device=0,source_sha256='test',identity={},max_samples=1)
        c.arm(); self.call(c)
        c.anchor_event.tick=0
        events=c.events[0]
        events[0].tick=600000.; events[1].tick=600001.
        for layer in range(43):
            events[2+2*layer].tick=600000.+layer*.02
            events[3+2*layer].tick=600000.+(layer+1)*.02
        sample=c.snapshot()['samples'][0]
        self.assertEqual(sample['routed'][0]['start_ms'],sample['routed'][0]['end_ms'])
        self.assertAlmostEqual(sample['routed'][0]['duration_ms'],.02,places=7)
        self.assertEqual(sample['gpu_duration_ms'],1.)

    def test_nonfinite_direct_duration_marks_collector_failed(self):
        b,c=self.make(max_samples=1)
        c.arm(); self.call(c)
        start=c.events[0][0]
        start.elapsed_time=lambda other: float('nan')
        report=c.snapshot()
        self.assertIsNone(report['samples'][0]['gpu_duration_ms'])
        self.assertIn('invalid_cuda_event_time',report['failures'])

    def test_idle_and_completeness(self):
        b,c = self.make()
        c.arm()
        with c.forward('prefill',1):
            with self.assertRaises(RuntimeError): c.snapshot()
        self.assertTrue(c.snapshot()['failures'])
    def test_overflow_explicit(self):
        b,c = self.make(max_calls=1,max_samples=1)
        c.arm(); self.call(c); self.call(c)
        r=c.snapshot()
        self.assertEqual(r['overflow'], {'calls':True,'samples':True})
        self.assertTrue(r['failures'])

class RequestTests(unittest.TestCase):
    def test_allowlist_missing_and_scheduler(self):
        from src.weave.request_timing import stall_timing_metadata
        class Stats:
            created_time=1.0
            first_token_time=0.0
            prompt='private'
        data=stall_timing_metadata(Stats(),None)
        self.assertEqual(data['created_time'],1.0)
        self.assertIn('first_token_time',data['missing'])
        self.assertNotIn('prompt',data)

class HookTests(unittest.TestCase):
    def test_lazy_native_and_custom_subclass_experts(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        from src.weave.stall_hooks import patch_model_class
        torch=ModuleType('torch')
        torch.cuda=SimpleNamespace(current_device=lambda:0)
        distributed=ModuleType('sglang.srt.distributed')
        distributed.get_tp_group=lambda:SimpleNamespace(rank=0)
        backend=Backend()
        invoked=[]
        class Expert:
            def forward(self): invoked.append('actual')
        class Native:
            def __init__(self): self.model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=Expert())) for _ in range(43)])
            def forward(self,ids,positions,batch):
                for layer in self.model.layers: layer.mlp.experts.forward()
                return 'native-result'
        patch_model_class(Native,'unused')
        class Custom(Native):
            def forward(self,*args):
                self.custom_installed=True
                return super().forward(*args)
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:False))
        with patch.dict(sys.modules,{'torch':torch,'sglang.srt.distributed':distributed}), patch('src.weave.stall_hooks.verify_source'), patch('src.weave.stall_hooks.start_endpoint'), patch('src.weave.stall_timeline.TorchBackend',return_value=backend):
            model=Custom()
            ids=SimpleNamespace(shape=[4])
            self.assertEqual(model.forward(ids,None,batch),'native-result')
            model._weave_stall_collector.arm()
            self.assertEqual(model.forward(ids,None,batch),'native-result')
            report=model._weave_stall_collector.snapshot()
            self.assertEqual(report['coverage']['prefill'],[1]*43)
            self.assertEqual(len(invoked),86)
            self.assertEqual(report['failures'],[])

class EndpointTests(unittest.TestCase):
    def test_local_control_idle_rules_and_source_checks(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        from urllib.request import Request,urlopen
        from urllib.error import HTTPError
        import json
        from src.weave.stall_hooks import start_endpoint,stall_source_sha256
        b=Backend()
        c=StallTimeline(b,rank=0,device=0,source_sha256=stall_source_sha256(),identity={})
        torch=SimpleNamespace(cuda=SimpleNamespace(device=lambda device:nullcontext()))
        with patch.dict('os.environ',{'WEAVE_STALL_PORT_BASE':'0'}),patch('src.weave.stall_hooks.verify_source'):
            server=start_endpoint(c,torch,'unused')
            base='http://127.0.0.1:'+str(server.server_address[1])
            try:
                with urlopen(Request(base+'/arm',data=b'{}'),timeout=2) as response:
                    self.assertTrue(json.load(response)['armed'])
                with c.forward('prefill',1):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base+'/snapshot',timeout=2)
                    self.assertEqual(error.exception.code,409)
                with urlopen(base+'/snapshot',timeout=2) as response:
                    self.assertTrue(json.load(response)['failures'])
                c.source_sha256='changed'
                with self.assertRaises(HTTPError): urlopen(base+'/status',timeout=2)
            finally:
                server.shutdown(); server.server_close()
