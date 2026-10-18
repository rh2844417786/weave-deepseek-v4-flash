"""CPU runtime/adapter contracts; CUDA execution is covered on the server."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests.cpu.test_staged_decode import Tensor
from tests.cpu.test_hopper_adapter import HopperAdapterTest
from tests.cpu.test_staged_adapter import Tensor as AdapterTensor
from tests.cpu.test_staged_launch import StagedLaunchTest
import json

class PrefillRuntimeTest(unittest.TestCase):
    def runtime(self, identity='a'*64):
        module=importlib.import_module('src.weave.hopper_prefill')
        class Workspace:
            def __init__(self,*dims):
                self.dimensions=dims; self.slot_outputs=Tensor(dims[2]*6,128); self.events=[]
                self.outputs={n:Tensor(n,128) for n in range(1,dims[2]+1)}
            def resources(self):return {'max_batch':8192}
            def bind_weights(self,raw):return SimpleNamespace(raw=raw,owner=self)
            def run(self,*args):self.events.append(('compute',args))
            def finalize(self,batch):self.events.append(('finalize',batch));return self.outputs[batch]
        dist=SimpleNamespace(ReduceOp=SimpleNamespace(SUM='SUM'))
        torch=SimpleNamespace(distributed=dist,cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))
        peer=SimpleNamespace(torch=torch,rank=1,ep=4,h=128,i=128,topk=6)
        loader=importlib.import_module('src.weave.hopper_prefill_candidate')
        with patch.object(loader,'extension',return_value=SimpleNamespace(PrefillWorkspace=Workspace,weave_source_sha256=identity)):
            runtime=module.HopperPrefillRuntime(peer,'group',experts=32)
        dist.all_reduce=lambda slots,op,group:runtime.ws.events.append(('sum',slots,op,group))
        return runtime
    def test_full_input_active_slots_order_and_cache(self):
        runtime=self.runtime(); raw=tuple(object() for _ in range(6)); bound=runtime.bind_weights(raw)
        x=Tensor(129,128);ids=Tensor(129,6);weights=Tensor(129,6)
        slices=runtime.ws.slot_outputs.slices
        first=runtime.run(x,ids,weights,bound);second=runtime.run(x,ids,weights,bound)
        self.assertIs(first,second);self.assertIs(bound.raw[0],raw[0]);self.assertEqual(runtime.ws.dimensions,(1,4,8192,128,128,32,6))
        self.assertEqual([e[0] for e in runtime.ws.events],['compute','sum','finalize']*2)
        self.assertIs(runtime.ws.events[0][1][0],x)
        self.assertEqual(runtime.ws.events[1][1].shape,(774,128));self.assertEqual(runtime.ws.events[1][2:],('SUM','group'))
        self.assertEqual(runtime.ws.slot_outputs.slices,slices);self.assertEqual(runtime.additional_slot_collectives,2)
        self.assertEqual(runtime.candidate_source_sha256,'a'*64)
        self.assertEqual(len(runtime.runtime_source_sha256),64)
    def test_invalid_batch_capture_and_identity(self):
        runtime=self.runtime()
        for batch in (0,8193):
            with self.assertRaisesRegex(ValueError,'8192'):runtime.run(Tensor(batch,128),None,None,None)
        runtime.torch.cuda.is_current_stream_capturing=lambda:True
        with self.assertRaisesRegex(RuntimeError,'eager'):runtime.run(Tensor(1,128),None,None,None)
        self.assertEqual(runtime.ws.events,[])
        with self.assertRaisesRegex(RuntimeError,'source identity'):self.runtime(None)
    def test_collective_failure_never_finalizes_or_counts(self):
        runtime=self.runtime()
        def fail(*args,**kwargs):raise RuntimeError('NCCL failed')
        runtime.torch.distributed.all_reduce=fail
        with self.assertRaisesRegex(RuntimeError,'NCCL failed'):
            runtime.run(Tensor(1,128),None,None,None)
        self.assertEqual([e[0] for e in runtime.ws.events],['compute'])
        self.assertEqual(runtime.additional_slot_collectives,0)
    def test_config_requires_hopper_decode(self):
        from src.weave.staged_decode import validate_prefill_configuration
        args=dict(prefill_implementation='hopper_fp8',decode_implementation='hopper_fp8',kernel='tiled',hot_cache=True,allow_decode_graph=False,fixed_plan=(1,1,0))
        validate_prefill_configuration(**args)
        for change in ({'decode_implementation':'persistent'},{'kernel':'scalar'},{'hot_cache':False},{'allow_decode_graph':True},{'fixed_plan':(0,1,0)},{'profile_path':'bank'},{'execution_profile':True},{'replay_capture':True},{'forward_profile':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_prefill_configuration(**{**args,**change})

class PrefillAdapterTest(unittest.TestCase):
    def test_prefill_full_inputs_single_mask_and_distinct_counts(self):
        module,expert,events,output=HopperAdapterTest().expert()
        prefill=SimpleNamespace(bind_weights=lambda raw:SimpleNamespace(raw=raw),run=lambda *a:events.append(('prefill',a)) or output)
        expert.runtime.hopper_prefill=prefill;expert.runtime.prefill_implementation='hopper_fp8';expert.runtime.phase='prefill'
        expert=module.RoutedExperts(SimpleNamespace(quant_method=object(),moe_runner_config=SimpleNamespace(inplace=True),should_fuse_routed_scaling_factor_in_topk=False),expert.runtime,expert.raw_weights,None,(1,1,0),10,5)
        x=AdapterTensor((8035,4096))
        with patch.object(module.torch,'zeros_like',side_effect=AssertionError('second mask')):
            result=expert.forward(x,SimpleNamespace(topk_ids=AdapterTensor((8035,6)),topk_weights=AdapterTensor((8035,6))))
        self.assertIs(result,output);self.assertIs(events[0][1][0],x);self.assertIs(events[0][1][3],expert.hopper_prefill_bound_weights)
        self.assertEqual(expert.implementation_calls['hopper_prefill'],1);self.assertEqual(expert.implementation_calls['hopper_decode'],0)

    def test_coverage_keeps_decode_and_prefill_collectives_separate(self):
        module,expert,_,_=HopperAdapterTest().expert()
        expert.runtime.hopper_decode.additional_slot_collectives=9
        expert.runtime.hopper_prefill=SimpleNamespace(candidate_source_sha256='d'*64,
            runtime_source_sha256='e'*64,resources={'max_batch':8192},additional_slot_collectives=5)
        expert.runtime.prefill_implementation='hopper_fp8'
        model=SimpleNamespace(weave_runtime=expert.runtime,model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(layer_id=5,experts=expert))]))
        report=module.coverage(model)
        self.assertEqual(report['additional_slot_collectives'],9)
        self.assertEqual(report['additional_prefill_slot_collectives'],5)
        self.assertEqual(report['prefill_source_sha256'],'d'*64)
        self.assertEqual(report['prefill_runtime_source_sha256'],'e'*64)
        self.assertEqual(report['candidate_source_sha256'],'a'*64)
        self.assertEqual(report['prefill_resources'],{'max_batch':8192})
        self.assertEqual(report['prefill_implementation'],'hopper_fp8')

class PrefillLaunchTest(StagedLaunchTest):
    def test_new_prefill_preview_and_stall_label(self):
        result=self.launch('--backend','weave_functional','--kernel','tiled','--decode-implementation','hopper_fp8','--prefill-implementation','hopper_fp8','--decode-graph','disabled','--stall-diagnostic','--request-timing')
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['docker_argv'];self.assertIn('WEAVE_PREFILL_IMPLEMENTATION=hopper_fp8',command);self.assertIn('WEAVE_STALL_CANDIDATE=hopper_prefill',command)
    def test_native_rejected(self):
        result=self.launch('--prefill-implementation','hopper_fp8')
        self.assertNotEqual(result.returncode,0);self.assertIn('hopper_fp8',result.stderr)
