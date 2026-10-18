import unittest
from pathlib import Path
from scripts.run_c32_comparison import service_command, gate_command
from src.weave.hopper_prefill_evidence import expected_cases, validate_prefill_reports

class PrefillLaunchTest(unittest.TestCase):
    def test_groups_share_decode_and_explicit_prefill(self):
        for variant in ('native','hopper_fp8','hopper_prefill'):
            command=service_command(variant,Path('/work'),Path('/model'),'test',30001,30110,'a'*64)
            self.assertEqual(command[command.index('--prefill-implementation')+1], 'hopper_fp8' if variant=='hopper_prefill' else 'persistent')
            self.assertEqual(command[command.index('--decode-implementation')+1], 'persistent' if variant=='native' else 'hopper_fp8')
        command=gate_command('sha256:image',Path('/work'),Path('/model'),'test','0,1,2,3','prefill',stall_diagnostic=True)
        self.assertIn('/workspace/scripts/run_prefill_gate.py',command)
        self.assertIn('--stall-diagnostic',command)
        self.assertIn('--model-path',command)

    def test_complete_source_bound_matrix_record_only(self):
        identity={k:'a'*64 for k in ('prefill_source_sha256','prefill_runtime_source_sha256','candidate_source_sha256','bf16_source_sha256','checkpoint_content_sha256')}
        identity['image_id']='sha256:image'
        rows=[]
        for rank in range(4):
            rows.append(dict(identity,rank=rank,world_size=4,status='passed',numerical_policy='record_only',cases=[dict(case,metadata_exact=True,finite=True,assembly_exact=True,stable_addresses=True,binding_alive=True,additional_slot_collectives=1,numerical={name:{'finite':True,'relative_l2':10.,'max_abs':99.} for name in ('old_fp8','fp8_oracle','bf16_oracle','bf16_reference')}) for case in expected_cases()]))
        self.assertEqual(validate_prefill_reports(rows,**identity)['status'],'passed')
        rows[0]['cases'].pop()
        with self.assertRaises(ValueError):validate_prefill_reports(rows,**identity)

    def test_prefill_coverage_uses_separate_collectives_and_all43(self):
        import copy
        from tests.cpu.test_c32_comparison import coverage
        from src.weave.c32_comparison import validate_coverage_delta
        before=[coverage(r,'hopper_fp8') for r in range(4)]
        after=[coverage(r,'hopper_fp8',True) for r in range(4)]
        for values,done in ((before,False),(after,True)):
            for row in values:
                row.update(candidate_source_sha256='c'*64,assembly_source_sha256='d'*64,numerical_policy='record_only',hopper_max_batch=32,hopper_slot_collective='torch.distributed.all_reduce/SUM/tp_device_group',prefill_implementation='hopper_fp8',prefill_source_sha256='e'*64,prefill_runtime_source_sha256='f'*64,additional_slot_collectives=86 if done else 0,additional_prefill_slot_collectives=43 if done else 0)
                for layer in row['layers'].values():
                    layer['device_calls'][0]=1 if done else 0
                    layer['implementation_calls'].update(hopper_decode=2 if done else 0,hopper_prefill=1 if done else 0,persistent_prefill=0)
        kwargs=dict(candidate_source_sha256='c'*64,assembly_source_sha256='d'*64,prefill_implementation='hopper_fp8',prefill_source_sha256='e'*64,prefill_runtime_source_sha256='f'*64)
        result=validate_coverage_delta(before,after,'hopper_fp8','a'*64,'b'*64,**kwargs)
        self.assertEqual(result['ranks']['0']['additional_prefill_slot_collectives'],43)
        bad=copy.deepcopy(before);del bad[0]['layers']['42']['implementation_calls']['hopper_prefill']
        with self.assertRaises(ValueError):validate_coverage_delta(bad,after,'hopper_fp8','a'*64,'b'*64,**kwargs)
        bad=copy.deepcopy(after);del bad[0]['layers']['42']['implementation_calls']['hopper_prefill']
        with self.assertRaises(ValueError):validate_coverage_delta(before,bad,'hopper_fp8','a'*64,'b'*64,**kwargs)
        bad=copy.deepcopy(after);bad[0]['additional_slot_collectives']+=43
        with self.assertRaises(ValueError):validate_coverage_delta(before,bad,'hopper_fp8','a'*64,'b'*64,**kwargs)
        bad=copy.deepcopy(after);bad[1]['layers']['42']['implementation_calls']['persistent_prefill']=1
        with self.assertRaises(ValueError):validate_coverage_delta(before,bad,'hopper_fp8','a'*64,'b'*64,**kwargs)

    def test_old_hopper_persistent_prefill_accepts_absent_hopper_prefill_counter(self):
        from tests.cpu.test_c32_comparison import coverage
        from src.weave.c32_comparison import validate_coverage_delta
        before=[coverage(r,'hopper_fp8') for r in range(4)]
        after=[coverage(r,'hopper_fp8',True) for r in range(4)]
        for values,done in ((before,False),(after,True)):
            for row in values:
                row.update(candidate_source_sha256='c'*64,assembly_source_sha256='d'*64,
                           numerical_policy='record_only',hopper_max_batch=32,
                           hopper_slot_collective='torch.distributed.all_reduce/SUM/tp_device_group',
                           prefill_implementation='persistent',additional_slot_collectives=86 if done else 0)
                for layer in row['layers'].values():
                    layer['device_calls'][0]=1 if done else 0
                    layer['implementation_calls'].update(hopper_decode=2 if done else 0,
                                                          persistent_prefill=1 if done else 0)
                    self.assertNotIn('hopper_prefill',layer['implementation_calls'])
        result=validate_coverage_delta(before,after,'hopper_fp8','a'*64,'b'*64,
            candidate_source_sha256='c'*64,assembly_source_sha256='d'*64,
            prefill_implementation='persistent')
        self.assertEqual(result['ranks']['0']['additional_slot_collectives'],86)
        self.assertEqual(result['ranks']['0']['additional_prefill_slot_collectives'],0)
        self.assertEqual(result['ranks']['0']['layers']['42']['implementation_delta']['persistent_prefill'],1)

    def test_prefill_wrapper_forces_suite_and_preserves_optional_stalls(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from scripts.run_prefill_comparison import main
        from src.weave.gpu_allocation import GPUAllocation
        base=['--work-root','/srv/work','--dataset','/srv/data.jsonl','--hopper-evidence-dir','/srv/proof']
        for extra in ([],['--comparison-suite=hopper'],['--comparison-suite','staged'],['--comparison-s=hopper','--stall-diagnostic']):
            output=io.StringIO()
            with patch('scripts.run_c32_comparison.allocated_gpus',return_value=GPUAllocation.parse('0,1,2,3')),contextlib.redirect_stdout(output):
                self.assertEqual(main(base+extra),0)
            value=json.loads(output.getvalue())
            self.assertEqual(value['comparison_suite'],'prefill')
            self.assertEqual(set(value['groups']),{'native','hopper_fp8','hopper_prefill'})
            if '--stall-diagnostic' in extra:self.assertIn('--stall-diagnostic',value['online_gate'])
