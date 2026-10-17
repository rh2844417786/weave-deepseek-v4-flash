"""Verify Hopper integration at the SGLang external module boundary."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests.cpu.test_staged_adapter import Tensor, backend_module
from tests.cpu.test_staged_launch import StagedLaunchTest
from src.weave.staged_decode import ModelBatchCounters


class HopperAdapterTest(unittest.TestCase):
    def expert(self):
        module=backend_module(); events=[]; output=Tensor((7,4096),'origin')
        hopper=SimpleNamespace(resources={'slot_bytes':100},additional_slot_collectives=0,
            candidate_source_sha256='a'*64,assembly_source_sha256='b'*64,
            bind_weights=lambda raw:SimpleNamespace(raw=raw),
            run=lambda *args:events.append(('hopper',args)) or output)
        runtime=SimpleNamespace(hot_cache=True,bind_weights=lambda raw:SimpleNamespace(raw=raw),
            staged_decode=None,hopper_decode=hopper,decode_implementation='hopper_fp8',device=0,ep=4,
            rank=2,phase='decode',allow_decode_graph=False,kernel='tiled',source_sha256='c'*64,
            fixed_plan=(1,1,0),trace_config=None,model_batch_counters=ModelBatchCounters(),
            run=lambda *args,**kwargs:events.append(('persistent',args,kwargs)) or Tensor((2,4096)))
        native=SimpleNamespace(quant_method=object(),moe_runner_config=SimpleNamespace(inplace=True),
                               should_fuse_routed_scaling_factor_in_topk=False)
        expert=module.RoutedExperts(native,runtime,tuple(object() for _ in range(6)),None,(1,1,0),10,5)
        return module,expert,events,output
    def test_full_replicated_decode_no_second_origin_mask_and_real_call_count(self):
        module,expert,events,output=self.expert(); x=Tensor((7,4096))
        with patch.object(module.torch,'zeros_like',side_effect=AssertionError('second mask')):
            result=expert.forward(x,SimpleNamespace(topk_ids=Tensor((7,6)),topk_weights=Tensor((7,6))))
        self.assertIs(result,output); self.assertIs(events[0][1][0],x)
        self.assertIs(events[0][1][3],expert.hopper_bound_weights)
        self.assertEqual(expert.implementation_calls['hopper_decode'],1)
        self.assertEqual(expert.implementation_calls['persistent_decode'],0)
    def test_prefill_uses_existing_bound_persistent_weights(self):
        _,expert,events,_=self.expert(); expert.runtime.phase='prefill'
        expert.forward(Tensor((7,4096)),SimpleNamespace(topk_ids=Tensor((7,6)),topk_weights=Tensor((7,6))))
        self.assertEqual(events[0][0],'persistent')
        self.assertIs(events[0][1][3],expert.bound_weights)
        self.assertEqual(expert.implementation_calls['persistent_prefill'],1)
    def test_coverage_carries_loaded_hopper_identities_and_actual_collectives(self):
        module,expert,_,_=self.expert(); expert.runtime.hopper_decode.additional_slot_collectives=7
        model=SimpleNamespace(weave_runtime=expert.runtime,model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(layer_id=5,experts=expert))]))
        report=module.coverage(model)
        self.assertEqual(report['candidate_source_sha256'],'a'*64)
        self.assertEqual(report['assembly_source_sha256'],'b'*64)
        self.assertEqual(report['additional_slot_collectives'],7)
        self.assertEqual(report['hopper_max_batch'],32)
        self.assertEqual(report['numerical_policy'],'record_only')
        self.assertIsNone(report['staged_resources'])


class HopperLaunchTest(StagedLaunchTest):
    def test_hopper_preview(self):
        result=self.launch('--backend','weave_functional','--kernel','tiled','--decode-implementation','hopper_fp8')
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['docker_argv']
        self.assertIn('WEAVE_DECODE_IMPLEMENTATION=hopper_fp8',command)
        self.assertEqual(command[command.index('--cuda-graph-backend-decode')+1],'disabled')
    def test_hopper_rejects_native_and_graph(self):
        for extra in (['--backend','native'],['--decode-graph','full']):
            result=self.launch('--backend','weave_functional','--kernel','tiled','--decode-implementation','hopper_fp8',*extra)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('hopper_fp8',result.stderr)

if __name__=='__main__':unittest.main()
