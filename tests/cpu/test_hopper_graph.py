"""CPU graph orchestration contract; this is not GPU replay evidence."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.weave import hopper_candidate, hopper_decode
from tests.cpu.test_hopper_decode import Workspace
from tests.cpu.test_staged_decode import Tensor


class Counter:
    device = 'cuda:0'
    value = 0
    def __init__(self, events): self.events = events
    def add_(self, value):
        self.events.append(('counter', value)); self.value += value
    def item(self):
        self.events.append(('item',)); return self.value


class HopperGraphTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('src.weave.hopper_graph')
        self.capture = False
        self.events = []
        self.counter = Counter(self.events)
        self.output = Tensor(32,128,'output')
        self.torch = SimpleNamespace(
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(SUM='SUM')),
            bfloat16='bf16', int64='int64', empty=lambda *a,**k:self.output,
            zeros=lambda *a,**k:self.counter,
            cuda=SimpleNamespace(is_current_stream_capturing=lambda:self.capture,
                                 synchronize=lambda device:self.events.append(('sync',device))))
        peer=SimpleNamespace(torch=self.torch,device=0,rank=2,ep=4,h=128,i=128,topk=6)
        ext=SimpleNamespace(FP8Workspace=Workspace,weave_source_sha256='a'*64)
        self.group=object()
        with patch.object(hopper_candidate,'extension',return_value=ext), patch.object(hopper_decode,'prepare_finalizer'):
            self.runtime=self.module.HopperGraphDecodeRuntime(peer,self.group,experts=32)
        self.runtime.ws.events=self.events
        self.torch.distributed.all_reduce=lambda tensor,op,group:self.events.append(('reduce',tensor,op,group))
        self.bound=self.runtime.bind_weights([object() for _ in range(6)])
        self.finalizer=patch.object(self.module,'finalize_slots',side_effect=lambda *a:self.events.append(('finalize',)+a))
        self.finalizer.start();self.addCleanup(self.finalizer.stop)

    def run_batch(self,batch):
        return self.runtime.run(Tensor(batch,128),Tensor(batch,6),Tensor(batch,6),self.bound,7.)

    def test_capture_requires_successful_eager_warmup_for_exact_batch(self):
        self.capture=True
        with self.assertRaisesRegex(RuntimeError,'warmup'):self.run_batch(7)
        self.assertEqual(self.events,[])
        self.capture=False;self.run_batch(7)
        self.capture=True;self.run_batch(7)
        with self.assertRaisesRegex(RuntimeError,'warmup'):self.run_batch(8)
        self.assertEqual(self.runtime.additional_slot_collectives,1)

    def test_full_input_cached_prefix_order_counter_and_no_host_reads(self):
        for batch in (32,1,17,7,32):
            result=self.run_batch(batch)
            self.assertIs(result,self.runtime._outputs[batch])
            compute,reduce,counter,finalize=self.events[-4:]
            self.assertEqual([item[0] for item in (compute,reduce,counter,finalize)],['compute','reduce','counter','finalize'])
            self.assertIs(reduce[1],self.runtime._slots[batch]);self.assertEqual(reduce[2:],('SUM',self.group))
            self.assertEqual(reduce[1].shape,(batch*6,128))
            self.assertEqual(counter,('counter',1))
            self.assertEqual(finalize[3:],(batch,128,6,2,4))
        self.assertEqual(self.output.slices,32);self.assertEqual(self.runtime.ws.slot_outputs.slices,32)
        self.assertFalse(any(x[0] in ('item','sync') for x in self.events))
        self.assertEqual(self.runtime.device_slot_collectives_snapshot(),5)
        self.assertEqual(self.events[-2:],[('sync','cuda:0'),('item',)])

    def test_failed_warmup_does_not_authorize_capture(self):
        with patch.object(self.module,'finalize_slots',side_effect=RuntimeError('finalizer')):
            with self.assertRaisesRegex(RuntimeError,'finalizer'):self.run_batch(1)
        self.capture=True
        with self.assertRaisesRegex(RuntimeError,'warmup'):self.run_batch(1)
        with self.assertRaisesRegex(RuntimeError,'capture'):self.runtime.device_slot_collectives_snapshot()

    def test_fake_graph_records_device_increment_and_replay_counts_without_python_run(self):
        operations=[]
        original=self.counter.add_
        def record(value):
            if self.capture:operations.append(lambda:original(value))
            else:original(value)
        self.counter.add_=record
        self.run_batch(7)
        self.capture=True;self.run_batch(7)
        self.assertEqual(self.counter.value,1)
        self.assertEqual(self.runtime.additional_slot_collectives,1)
        self.capture=False
        before_compute=len([event for event in self.events if event[0]=='compute'])
        for _ in range(3):
            for operation in operations:operation()
        self.assertEqual(self.runtime.device_slot_collectives_snapshot(),4)
        self.assertEqual(self.runtime.additional_slot_collectives,1)
        self.assertEqual(len([event for event in self.events if event[0]=='compute']),before_compute)

    def test_invalid_batch_fails_before_compute(self):
        for batch in (0,33):
            with self.assertRaises(ValueError):self.run_batch(batch)
        self.assertEqual(self.events,[])

    def test_source_identity_and_collective_failure(self):
        self.assertEqual(self.runtime.graph_runtime_source_sha256,self.module.graph_runtime_source_sha256())
        self.assertEqual(self.runtime.assembly_source_sha256,hopper_decode.assembly_source_sha256())
        def fail(*a,**k):raise RuntimeError('reduce')
        self.torch.distributed.all_reduce=fail
        with self.assertRaisesRegex(RuntimeError,'reduce'):self.run_batch(1)
        self.assertEqual([item[0] for item in self.events],['compute'])
        self.assertEqual(self.counter.value,0)



class GraphSmokeContractTest(unittest.TestCase):
    def test_exact_matrix_and_source_identity_without_gpu_import(self):
        from tests.gpu.check_hopper_graph import CAPTURE_BATCHES, expected_cases, hashes
        self.assertEqual(CAPTURE_BATCHES,tuple(range(32,0,-1)))
        cases=expected_cases()
        self.assertEqual(len(cases),38)
        self.assertEqual([case['batch'] for case in cases[:32]],list(range(1,33)))
        self.assertEqual([case['batch'] for case in cases[32:37]],[32,1,17,7,32])
        self.assertEqual([case['handle'] for case in cases[32:37]],[0,1,0,1,0])
        self.assertEqual(cases[-1],dict(case='after_prefill',batch=32,handle=0,variant=9))
        self.assertTrue(all(len(value)==64 for value in hashes().values()))

    def capture_case(self,fail_compute=False):
        from tests.gpu import check_hopper_graph as check
        events=[]
        class Context:
            def __enter__(self):events.append('capture_enter')
            def __exit__(self,*args):events.append('capture_exit')
        dist=SimpleNamespace(ReduceOp=SimpleNamespace(SUM='SUM'),
            all_reduce=lambda *a,**kw:events.append('nccl'))
        torch=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=lambda:object(),
            synchronize=lambda:events.append('sync'),graph=lambda *a,**kw:Context()))
        output=object();slots=object();group=object()
        def run(*a):
            events.append('compute')
            if fail_compute:raise RuntimeError('before NCCL')
            dist.all_reduce(slots,op='SUM',group=group)
            events.append('counter_finalize');return output
        runtime=SimpleNamespace(_slots={7:slots},device_group=group,run=run)
        def agreed(label,operation):
            events.append(label)
            try:return operation()
            except Exception:raise check.smoke.AgreedFailure(label)
        patches=[patch.object(check,'torch',torch,create=True),patch.object(check,'dist',dist,create=True),
                 patch.object(check.smoke,'agreed',side_effect=agreed),patch.object(check.smoke,'private_exception')]
        for item in patches:item.start();self.addCleanup(item.stop)
        return check,events,runtime,output,dist

    def test_capture_agrees_before_nccl_and_never_synchronizes_inside_body(self):
        check,events,runtime,output,dist=self.capture_case()
        original=dist.all_reduce
        graph,result=check.capture(runtime,(Tensor(7,128),None,None),object(),object(),'case')
        self.assertIs(result,output);self.assertIs(dist.all_reduce,original)
        self.assertLess(events.index('case:pre_collective'),events.index('nccl'))
        self.assertFalse('sync' in events[events.index('capture_enter'):events.index('capture_exit')])

    def test_failed_capture_retains_graph_for_explicit_cleanup(self):
        check,events,runtime,output,dist=self.capture_case(fail_compute=True)
        owned=[]
        with self.assertRaises(check.smoke.AgreedFailure):
            check.capture(runtime,(Tensor(7,128),None,None),object(),object(),'case',owned_graphs=owned)
        self.assertEqual(len(owned),1)

    def test_pre_nccl_local_failure_joins_same_agreement_and_restores_reduce(self):
        check,events,runtime,output,dist=self.capture_case(fail_compute=True)
        original=dist.all_reduce
        with self.assertRaises(check.smoke.AgreedFailure):
            check.capture(runtime,(Tensor(7,128),None,None),object(),object(),'case')
        self.assertIn('case:pre_collective',events);self.assertNotIn('nccl',events)
        self.assertIs(dist.all_reduce,original)


class GraphReplayFreshnessTest(unittest.TestCase):
    def test_counter_only_replay_cannot_inherit_eager_metadata(self):
        from tests.gpu import check_hopper_graph as check
        class Buffer:
            def __init__(self,values):self.values=values
            def fill_(self,value):
                self.values=[value for _ in self.values];return self
            def cpu(self):return self
            def tolist(self):return self.values
            def copy_(self,value):return self
        ids=Buffer([[0,1,2,3,4,5]])
        ws=SimpleNamespace(group_counts=Buffer([1,1,1,1,1,1,0,0]),
            active_count=Buffer([6]),active_experts=Buffer([0,1,2,3,4,5,-1,-1]),
            expert_slots=Buffer([[0]+[-1]*31 for _ in range(8)]))
        counter=[0]
        def replay():counter[0]+=1
        runtime=SimpleNamespace(ws=ws,_slots={1:Buffer([0])},_outputs={1:Buffer([0])},
            device_slot_collectives_snapshot=lambda:counter[0])
        graph=SimpleNamespace(replay=replay)
        with patch.object(check,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)),create=True), \
             patch.object(check.smoke,'agreed',side_effect=lambda label,operation:operation()), \
             patch.object(check.smoke,'addresses',return_value='stable'), \
             patch.object(check.smoke,'synthetic_inputs',return_value=(Buffer([0]),ids,Buffer([0]))), \
             patch.object(check.smoke,'run_captured') as eager:
            with self.assertRaisesRegex(AssertionError,'group counts'):
                check.evaluate(runtime,None,None,None,None,(Buffer([0]),ids,Buffer([0])),graph,
                    dict(batch=1,handle=0,variant=1,case='after_prefill'),'stable',0)
            eager.assert_not_called()
            self.assertEqual(counter[0],1)
            self.assertEqual(ws.group_counts.tolist(),[-2]*8)


if __name__=='__main__':unittest.main()
