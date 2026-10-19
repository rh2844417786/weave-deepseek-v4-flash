"""CPU verification of paper profile identity and actual routed adapter selection."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.weave.hopper_persistent import (HopperPersistentRuntime, IMPLEMENTATION,
    validate_profile_bank, source_manifest)
from src.weave.ops import ProfileBank
from src.weave.staged_decode import (ModelBatchCounters, validate_decode_configuration,
    validate_prefill_configuration)
from tests.cpu.test_staged_adapter import Tensor, backend_module
from tests.cpu.test_staged_launch import StagedLaunchTest


class ProfileTest(unittest.TestCase):
    def fixture(self):
        expected = dict(source_sha256='a'*64, device_key='H100;sm90;132',hidden_size=128,
            intermediate_size=128, experts=32, top_k=2, rank=0, workers=132, capacity=32,
            image_id='sha256:'+'b'*64, checkpoint_content_sha256='c'*64,
            runtime={'python':'3.12','torch':'2','cuda':'13'})
        data = {k:v for k,v in expected.items() if k not in ('workers','capacity')}
        data.update(schema_version=1,status='measured',implementation=IMPLEMENTATION,
            numerical_policy='record_only',model_constants={'efficiency':1.,'overlap':0.},
            profiles=[{'max_local_tokens':32,'candidates':[[1,1,1e6,1e8,1.,0.],[2,2,2e6,2e8,1.,0.]]}])
        return data, expected
    def test_measured_bank_with_all_identity_dimensions(self):
        data, expected = self.fixture()
        self.assertEqual(validate_profile_bank(data, **expected), data['profiles'])
        for key in ('implementation','source_sha256','device_key','hidden_size','intermediate_size',
                    'rank','image_id','checkpoint_content_sha256','runtime','experts','top_k'):
            bad=copy.deepcopy(data);bad[key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):validate_profile_bank(bad,**expected)
    def test_source_manifest_isolates_new_kernel_and_hashes_actual_shared_headers(self):
        manifest=source_manifest()
        files={r['path'] for r in manifest['files']}
        self.assertIn('experimental/hopper_persistent/megakernel.cu',files)
        self.assertIn('experimental/hopper_fp8/math_contract.h',files)
        self.assertIn('csrc/expert_numerics.h',files)
        self.assertNotIn('csrc/megakernel.cu',files)
        self.assertNotIn('experimental/hopper_fp8/kernel.cu',files)
        self.assertNotIn('experimental/hopper_fp8/prefill.cu',files)
        self.assertEqual(manifest['implementation'],IMPLEMENTATION)
        self.assertTrue(all(len(r['sha256'])==64 for r in manifest['files']))

    def test_reject_incomplete_coverage_or_unmeasured_selector(self):
        for update in ('short','no_alternatives','nan','negative','constant'):
            data, expected = self.fixture(); b=data['profiles'][0]
            if update=='short':b['max_local_tokens']=8
            elif update=='no_alternatives':b['candidates'][1][1]=1
            elif update=='nan':b['candidates'][0][2]=float('nan')
            elif update=='negative':b['candidates'][0][3]=-1
            else:b['candidates'][0][5]=0.5
            with self.subTest(update=update),self.assertRaises(ValueError):validate_profile_bank(data,**expected)


class ConfigurationTest(unittest.TestCase):
    def test_paired_eager_or_full_graph_with_measured_profile(self):
        base=dict(decode_implementation='hopper_persistent',kernel='tiled',hot_cache=True,
            allow_decode_graph=False,fixed_plan=(1,1,0),profile_path='/bank')
        validate_decode_configuration(**base)
        validate_prefill_configuration(prefill_implementation='hopper_persistent',**base)
        validate_decode_configuration(**{**base,'allow_decode_graph':True,'graph_c32':True})
        for extra in ({'profile_path':None},{'execution_profile':True},{'replay_capture':True},
                      {'forward_profile':True},{'allow_decode_graph':True},{'hot_cache':False}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_decode_configuration(**{**base,**extra})
        with self.assertRaises(ValueError):validate_prefill_configuration(prefill_implementation='persistent',**base)


class AdapterTest(unittest.TestCase):
    def test_all_layers_both_phases_use_paper_bound_runtime_and_origin_mask(self):
        module=backend_module(); calls=[]; profile=object()
        runtime=SimpleNamespace(implementation=IMPLEMENTATION,hot_cache=True,
            bind_weights=lambda raw:SimpleNamespace(raw=raw),staged_decode=None,hopper_decode=None,
            hopper_prefill=None,decode_implementation='hopper_persistent',
            prefill_implementation='hopper_persistent',device=0,ep=4,rank=2,
            phase='decode',allow_decode_graph=True,graph_c32=True,
            run=lambda *args,**kwargs:calls.append((args,kwargs)) or Tensor((2,4096)))
        native=SimpleNamespace(quant_method=object(),moe_runner_config=SimpleNamespace(inplace=True),
            should_fuse_routed_scaling_factor_in_topk=False)
        experts=[module.RoutedExperts(native,runtime,tuple(object() for _ in range(6)),
            profile,(1,1,0),10,layer) for layer in range(43)]
        for phase in ('prefill','decode'):
            runtime.phase=phase
            for expert in experts:
                result=expert.forward(Tensor((7,4096)),SimpleNamespace(
                    topk_ids=Tensor((7,6)),topk_weights=Tensor((7,6))))
                self.assertEqual(result.label,'masked')
                self.assertEqual(calls[-1][0][0].shape,(2,4096))
                self.assertIs(calls[-1][0][3],expert.bound_weights)
                self.assertIs(calls[-1][1]['profile'],profile)
                self.assertEqual(calls[-1][1]['trace_layer'],expert.layer_id)
                self.assertEqual(expert.implementation_calls['paper_persistent_'+phase],1)
                self.assertEqual(expert.implementation_calls['persistent_'+phase],0)
        self.assertEqual(len(calls),86)
        runtime.phase='decode'
        with patch.object(module.torch.cuda,'is_current_stream_capturing',return_value=True):
            experts[0].forward(Tensor((32,4096)),SimpleNamespace(
                topk_ids=Tensor((32,6)),topk_weights=Tensor((32,6))))
        self.assertEqual(experts[0].capture_calls,1)


    def test_paper_graph_coverage_has_own_evidence_without_operator_slot_runtime(self):
        module=backend_module(); profile=SimpleNamespace(sha256='p'*64)
        runtime=SimpleNamespace(implementation=IMPLEMENTATION,rank=0,kernel='tiled',
            source_sha256='a'*64,profile_bank_sha256='p'*64,resources={'one_cta_per_sm':True},
            allow_decode_graph=True,graph_c32=True,trace_config=None,hot_cache=True,
            decode_implementation='hopper_persistent',prefill_implementation='hopper_persistent',
            staged_decode=None,hopper_decode=None,hopper_prefill=None,
            model_batch_counters=ModelBatchCounters(),mechanism_snapshot=lambda:{'phases':{}})
        expert=SimpleNamespace(profile=profile,device_calls=Tensor((3,)),capture_calls=32,
            coverage={'prefill':1,'decode':1,'other':0},
            implementation_calls={'paper_persistent_decode':1,'paper_persistent_prefill':1})
        model=SimpleNamespace(weave_runtime=runtime,model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(layer_id=i,experts=expert)) for i in range(43)]))
        report=module.coverage(model)
        self.assertEqual(report['method_identity'],IMPLEMENTATION)
        self.assertEqual(report['method_status'],'paper_candidate')
        self.assertEqual(report['numerical_policy'],'record_only')
        self.assertEqual(report['additional_slot_collectives'],0)
        self.assertEqual(len(report['layers']),43)
        self.assertEqual(report['mechanism_evidence'],{'phases':{}})
        self.assertEqual(report['graph_mode'],'full')


class RuntimeTensor:
    def __init__(self, shape, dtype='bf16',device=None):
        self.shape=shape; self.ndim=len(shape);self.dtype=dtype
        self.device=device or SimpleNamespace(index=0);self.events=[]
    def __getitem__(self,key):return self
    def copy_(self,value):self.events.append(('copy',value))
    def fill_(self,value):self.events.append(('fill',value))
    def to(self,dtype):return self


class RuntimeTest(unittest.TestCase):
    def test_dynamic_layer_evidence_without_legacy_trace_and_no_host_reads(self):
        runtime=object.__new__(HopperPersistentRuntime)
        runtime.torch=SimpleNamespace(bfloat16='bf16',int32='i32',float32='f32')
        runtime.h=128;runtime.topk=2;runtime.capacity=32;runtime.device=0;runtime.phase='decode'
        runtime.trace_config=None;runtime.empty_profile=object();calls=[]
        runtime.ws=SimpleNamespace(run_bound=lambda *args:calls.append(args))
        runtime._views=tuple(RuntimeTensor((32,128)) for _ in range(5))
        device=SimpleNamespace(index=0)
        x=RuntimeTensor((8,128),device=device);ids=RuntimeTensor((8,2),device=device)
        weights=RuntimeTensor((8,2),device=device); tensor=object()
        bank=ProfileBank([(32,tensor)]);bound=object()
        runtime.run(x,ids,weights,bound,profile=bank,trace_layer=42)
        self.assertIs(calls[0][1],tensor)
        self.assertEqual(calls[0][2:5],(-1,1,0))
        self.assertEqual(calls[0][-4:],(2,0,42,1))
        runtime.run(x,ids,weights,bound,fixed_plan=(2,4,8),calibration_mode='communication')
        self.assertEqual(calls[1][2:5],(2,4,8))
        self.assertEqual(calls[1][-4:],(2,1,-1,1))
        with self.assertRaisesRegex(ValueError,'shape-defined'):
            runtime.run(x,ids,weights,bound,valid_count=object())


class PaperLaunchTest(StagedLaunchTest):
    def test_graph_profile_is_mounted_and_instrumentation_rejected(self):
        base=['--backend','weave_functional','--kernel','tiled',
            '--decode-implementation','hopper_persistent','--prefill-implementation','hopper_persistent',
            '--work-root','/tmp/paper-work','--profile','/tmp/paper-work/banks',
            '--checkpoint-content-sha256','a'*64]
        for graph in ([],['--graph-c32','--decode-graph','full','--request-timing']):
            result=self.launch(*base,*graph,environment={'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'})
            self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads(result.stdout)['docker_argv']
            self.assertIn('WEAVE_PROFILE=/work/banks',argv)
            self.assertIn('WEAVE_DECODE_IMPLEMENTATION=hopper_persistent',argv)
            self.assertIn('--pull=never',argv)
        result=self.launch(*base,'--graph-c32','--decode-graph','full','--request-timing',
            '--graph-stage-profile',environment={'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'})
        self.assertNotEqual(result.returncode,0)
        self.assertIn('legacy graph stage',result.stderr)

if __name__=='__main__':unittest.main()
