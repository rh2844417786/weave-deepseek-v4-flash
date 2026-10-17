"""CPU contract checks for actual Hopper orchestration; no Torch installation."""
import importlib
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
from src.weave import hopper_candidate
from tests.cpu.test_staged_decode import Tensor


class Workspace:
    def __init__(self, *dimensions):
        self.dimensions = dimensions
        self.slot_outputs = Tensor(192, dimensions[3], 'slots')
        self.events = []
    def resources(self):
        return {'max_batch': 32, 'slot_bytes': 192 * 128 * 4}
    def bind_weights(self, raw):
        return types.SimpleNamespace(owner=self, raw=raw)
    def run(self, x, ids, routing, bound, clamp):
        if bound.owner is not self:
            raise ValueError('foreign workspace')
        self.events.append(('compute', x, ids, routing, bound, clamp))


class HopperDecodeTest(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('src.weave.hopper_decode')
        except ModuleNotFoundError:
            self.fail('Hopper serving runtime is missing')
    def runtime(self, identity='a'*64):
        module = self.module()
        output = Tensor(32, 128, 'output')
        dist = types.SimpleNamespace(ReduceOp=types.SimpleNamespace(SUM='SUM'))
        torch = types.SimpleNamespace(distributed=dist, bfloat16='bf16',
            empty=lambda *a, **kw: output,
            cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
        peer = types.SimpleNamespace(torch=torch, device=0, rank=2, ep=4, h=128, i=128, topk=6)
        extension = types.SimpleNamespace(FP8Workspace=Workspace, weave_source_sha256=identity)
        group = object()
        with patch.object(hopper_candidate, 'extension', return_value=extension), patch.object(module, 'prepare_finalizer'):
            runtime = module.HopperDecodeRuntime(peer, group, experts=32)
        dist.all_reduce=lambda tensor, op, group: runtime.ws.events.append(('reduce',tensor,op,group))
        return runtime, group, output
    def test_full_inputs_one_active_prefix_sum_then_finalize(self):
        runtime, group, output = self.runtime()
        raw=tuple(object() for _ in range(6)); bound=runtime.bind_weights(raw)
        x, ids, routing=Tensor(7,128), Tensor(7,6), Tensor(7,6)
        def finalize(*args): runtime.ws.events.append(('finalize',)+args)
        with patch.object(self.module(),'finalize_slots',side_effect=finalize):
            result=runtime.run(x,ids,routing,bound,10)
        events=runtime.ws.events
        self.assertEqual([e[0] for e in events],['compute','reduce','finalize'])
        self.assertIs(events[0][1],x)
        self.assertEqual(events[1][1].shape,(42,128))
        self.assertEqual(events[1][2:],('SUM',group))
        self.assertEqual(events[2][3:],(7,128,6,2,4))
        self.assertIs(events[2][2],output)
        self.assertIs(result.owner,output)
        self.assertEqual(runtime.additional_slot_collectives,1)
        self.assertEqual(runtime.candidate_source_sha256,'a'*64)
        self.assertEqual(runtime.assembly_source_sha256,self.module().assembly_source_sha256())
    def test_cached_views_and_weight_lifetimes(self):
        runtime,_,output=self.runtime()
        self.assertEqual(runtime.ws.dimensions,(2,4,32,128,128,32,6))
        bound=runtime.bind_weights(tuple(object() for _ in range(6)))
        reads=(runtime.ws.slot_outputs.slices,output.slices)
        with patch.object(self.module(),'finalize_slots'):
            a=runtime.run(Tensor(32,128),Tensor(32,6),Tensor(32,6),bound)
            runtime.run(Tensor(1,128),Tensor(1,6),Tensor(1,6),bound)
            b=runtime.run(Tensor(32,128),Tensor(32,6),Tensor(32,6),bound)
        self.assertIs(a,b)
        self.assertEqual(reads,(32,32))
        self.assertEqual(reads,(runtime.ws.slot_outputs.slices,output.slices))
        self.assertIs(bound.owner,runtime.ws)
    def test_fail_closed_before_collective(self):
        runtime,_,_=self.runtime(); bound=runtime.bind_weights([])
        for batch in (0,33):
            with self.assertRaisesRegex(ValueError,'1..32'):
                runtime.run(Tensor(batch,128),None,None,bound)
        runtime.torch.cuda.is_current_stream_capturing=lambda:True
        with self.assertRaisesRegex(RuntimeError,'eager'):
            runtime.run(Tensor(1,128),None,None,bound)
        self.assertEqual(runtime.ws.events,[])
    def test_compute_or_collective_failure_never_finalizes(self):
        runtime,_,_=self.runtime(); bound=runtime.bind_weights([])
        def fail(*args,**kwargs): raise RuntimeError('NCCL failed')
        runtime.torch.distributed.all_reduce=fail
        with patch.object(self.module(),'finalize_slots') as finalize:
            with self.assertRaisesRegex(RuntimeError,'NCCL failed'):
                runtime.run(Tensor(1,128),None,None,bound)
            finalize.assert_not_called()
        self.assertEqual(runtime.additional_slot_collectives,0)
    def test_missing_loaded_source_identity_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'source identity'):
            self.runtime(identity=None)
    def test_configuration_rejects_unsupported_hopper_modes(self):
        from src.weave.staged_decode import validate_decode_configuration
        args=dict(decode_implementation='hopper_fp8',kernel='tiled',hot_cache=True,
                  allow_decode_graph=False,fixed_plan=(1,1,0))
        validate_decode_configuration(**args)
        for change in ({'kernel':'scalar'},{'hot_cache':False},{'allow_decode_graph':True},
                       {'fixed_plan':(0,1,0)},{'profile_path':'bank'},
                       {'execution_profile':True},{'replay_capture':True},{'forward_profile':True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_decode_configuration(**{**args,**change})
    def test_import_requires_no_torch_or_triton(self):
        script="import sys; import src.weave.hopper_decode, src.weave.hopper_finalize; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__': unittest.main()
