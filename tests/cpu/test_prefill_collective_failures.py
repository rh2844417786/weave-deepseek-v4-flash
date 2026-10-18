"""Inject runtime faults without Torch/CUDA to verify collective phase ordering."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests.gpu import check_hopper_prefill as check

class PrefillCollectiveFailureTest(unittest.TestCase):
    def invoke(self,kind):
        phases=[];nccl=[];sync=[]
        tensor=object();group=object()
        def synchronize():
            sync.append('sync')
            raise RuntimeError('injected device synchronization failure')
        distributed=SimpleNamespace(all_reduce=lambda *a,**kw:nccl.append('SUM'),ReduceOp=SimpleNamespace(SUM='SUM'))
        runtime=SimpleNamespace(additional_slot_collectives=0,device_group=group,_slots={1:tensor})
        def run(*args):
            distributed.all_reduce(tensor,op='SUM',group=group)
            if kind=='duplicate':distributed.all_reduce(tensor,op='SUM',group=group)
        runtime.run=run
        def agreed(label,operation):
            phases.append(label)
            # Duplicate-call test skips tensor-heavy checks after the first entry,
            # then insists runtime completion sees the second-call exception.
            if kind=='duplicate' and not label.endswith(':complete'):return None
            try:return operation()
            except Exception:raise check.smoke.AgreedFailure(label)
        fake_checkpoint=SimpleNamespace(expert_reference=lambda *args:None)
        with patch.dict(sys.modules,{'tests.gpu.check_checkpoint':fake_checkpoint}),patch.object(check,'addresses',return_value=()),patch.object(check,'dist',distributed,create=True),patch.object(check,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)),create=True),patch.object(check.smoke,'agreed',side_effect=agreed),patch.object(check.smoke,'private_exception'):
            with self.assertRaises(check.smoke.AgreedFailure):
                check.evaluate(runtime,None,None,None,None,None,None,(SimpleNamespace(shape=(1,128)),None,None),0,32,7.,'fault')
        return phases,nccl,sync

    def test_sync_failure_joins_precollective_then_common_completion(self):
        phases,nccl,sync=self.invoke('sync')
        self.assertEqual(phases,['fault:pre_collective','fault:complete'])
        self.assertEqual(nccl,[])
        self.assertEqual(sync,['sync'])

    def test_second_collective_rejects_before_second_precollective(self):
        phases,nccl,sync=self.invoke('duplicate')
        self.assertEqual(phases,['fault:pre_collective','fault:summed_slots','fault:complete'])
        self.assertEqual(nccl,['SUM'])
        self.assertEqual(sync,[])
