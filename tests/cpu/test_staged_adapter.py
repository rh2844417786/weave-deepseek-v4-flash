"""Exercise actual routed-expert and model wrappers at the external CUDA boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.weave.staged_decode import ModelBatchCounters

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, label='tensor'):
        self.shape, self.label, self.copies = shape, label, []
    def __getitem__(self, key):
        if isinstance(key,int):return self
        return Tensor((key.stop-(key.start or 0),*self.shape[1:]),self.label)
    def copy_(self,value):self.copies.append(value);return self
    def add_(self,value):return self
    def cpu(self):return self
    def tolist(self):return [0,0,0]


class Module:
    def register_buffer(self,name,value):setattr(self,name,value)


def backend_module():
    torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),int64='i64',
        cuda=SimpleNamespace(is_current_stream_capturing=lambda:False),
        zeros=lambda shape,**kwargs:Tensor((shape,) if isinstance(shape,int) else shape),
        zeros_like=lambda value:Tensor(value.shape,'masked'))
    spec=importlib.util.spec_from_file_location('src.weave.test_staged_backend',ROOT/'src/weave/sglang_backend.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'torch':torch}):spec.loader.exec_module(module)
    return module


class StagedAdapterTest(unittest.TestCase):
    def expert(self,implementation='staged_direct'):
        module=backend_module();events=[]
        output=Tensor((7,4096),'already-origin-masked')
        staged=SimpleNamespace(resources={},bind_weights=lambda raw:SimpleNamespace(raw=raw),
            run=lambda *args:events.append(('staged',args)) or output)
        runtime=SimpleNamespace(hot_cache=True,bind_weights=lambda raw:SimpleNamespace(raw=raw),
            staged_decode=staged if implementation=='staged_direct' else None,
            decode_implementation=implementation,device=0,ep=4,rank=2,
            phase='decode',allow_decode_graph=False,
            run=lambda *args,**kwargs:events.append(('persistent',args,kwargs)) or Tensor((2,4096)))
        native=SimpleNamespace(quant_method=object(),moe_runner_config=SimpleNamespace(inplace=True),
                               should_fuse_routed_scaling_factor_in_topk=False)
        expert=module.RoutedExperts(native,runtime,tuple(object() for _ in range(6)),None,(1,1,0),10,5)
        return module,expert,events,output

    def test_staged_decode_keeps_full_replicated_batch_and_does_not_mask_twice(self):
        module,expert,events,output=self.expert()
        x=Tensor((7,4096));ids=Tensor((7,6));routing=Tensor((7,6))
        with patch.object(module.torch,'zeros_like',side_effect=AssertionError('second mask')):
            result=expert.forward(x,SimpleNamespace(topk_ids=ids,topk_weights=routing))
        self.assertIs(result,output)
        self.assertEqual([event[0] for event in events],['staged'])
        self.assertIs(events[0][1][0],x)
        self.assertEqual(expert.implementation_calls,{'staged_decode':1,'persistent_decode':0,
            'persistent_prefill':0,'persistent_other':0})

    def test_prefill_retains_cached_persistent_weights_and_contiguous_origin_partition(self):
        _,expert,events,_=self.expert();expert.runtime.phase='prefill'
        result=expert.forward(Tensor((7,4096)),SimpleNamespace(topk_ids=Tensor((7,6)),topk_weights=Tensor((7,6))))
        self.assertEqual(result.label,'masked')
        self.assertEqual(events[0][0],'persistent')
        self.assertEqual(events[0][1][0].shape,(2,4096))
        self.assertIs(events[0][1][3],expert.bound_weights)
        self.assertEqual(expert.implementation_calls['persistent_prefill'],1)
        self.assertEqual(expert.implementation_calls['staged_decode'],0)

    def test_persistent_decode_control_remains_the_default_counted_branch(self):
        _,expert,events,_=self.expert('persistent')
        expert.forward(Tensor((7,4096)),SimpleNamespace(topk_ids=Tensor((7,6)),topk_weights=Tensor((7,6))))
        self.assertEqual(events[0][0],'persistent')
        self.assertEqual(expert.implementation_calls['persistent_decode'],1)

    def test_coverage_rejects_active_and_interleaved_model_forwards(self):
        module,expert,_,_=self.expert()
        runtime=expert.runtime
        runtime.kernel='tiled';runtime.source_sha256='source';runtime.fixed_plan=(1,1,0)
        runtime.trace_config=None;runtime.model_batch_counters=ModelBatchCounters()
        runtime.staged_decode.additional_slot_collectives=0
        model=SimpleNamespace(weave_runtime=runtime,model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(layer_id=5,experts=expert))]))
        runtime.model_batch_counters.begin('decode',7,7)
        with self.assertRaisesRegex(ValueError,'active'):module.coverage(model)
        runtime.model_batch_counters.end(success=True)
        before=module.coverage(model)
        self.assertEqual(before['model_batch_counts']['by_phase']['decode'],{'7':1})
        self.assertEqual(before['decode_implementation'],'staged_direct')
        def interleaved_cpu():
            runtime.model_batch_counters.begin('decode',1,1)
            runtime.model_batch_counters.end(success=True)
            return expert.device_calls
        expert.device_calls.cpu=interleaved_cpu
        with self.assertRaisesRegex(ValueError,'active'):module.coverage(model)


class NativeModel:
    def forward(self,*args,**kwargs):
        if getattr(self,'fail',False):raise RuntimeError('native failed')
        return 'output'


class ModelBoundaryTest(unittest.TestCase):
    def model(self):
        modules={'sglang.srt.models.deepseek_v4':SimpleNamespace(DeepseekV4ForCausalLM=NativeModel),
                 'weave.sglang_backend':SimpleNamespace(install_weave_backend=None,coverage=None)}
        spec=importlib.util.spec_from_file_location('test_staged_model',ROOT/'src/weave/sglang_ext/models/deepseek_v4.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',modules):spec.loader.exec_module(module)
        model=module.DeepseekV4ForCausalLM()
        model.weave_runtime=SimpleNamespace(model_batch_counters=ModelBatchCounters())
        return model

    def test_unprofiled_eager_forward_records_cpu_shapes_and_failure_without_sampler(self):
        model=self.model()
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:True,is_extend=lambda:False),batch_size=7)
        self.assertEqual(model.forward(input_ids=Tensor((7,)),forward_batch=batch),'output')
        model.fail=True
        with self.assertRaisesRegex(RuntimeError,'native failed'):
            model.forward(input_ids=Tensor((7,)),forward_batch=batch)
        snapshot=model.weave_runtime.model_batch_counters.snapshot()
        self.assertEqual(snapshot['by_phase']['decode'],{'7':1})
        self.assertEqual(snapshot['failed_forwards'],1)
        self.assertFalse(snapshot['active'])

    def test_diagnostic_initialization_failure_releases_model_coverage(self):
        model=self.model()
        def fail(*args):raise RuntimeError('sampler failed')
        model.weave_runtime.forward_sampler=SimpleNamespace(begin=fail)
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:True,is_extend=lambda:False),batch_size=7)
        with self.assertRaisesRegex(RuntimeError,'sampler failed'):
            model.forward(input_ids=Tensor((7,)),forward_batch=batch)
        try:snapshot=model.weave_runtime.model_batch_counters.snapshot()
        except ValueError:self.fail('diagnostic failure leaves model coverage active')
        self.assertEqual(snapshot['failed_forwards'],1)


if __name__=='__main__':unittest.main()
