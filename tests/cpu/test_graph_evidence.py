import copy
import unittest
from unittest.mock import patch
import tempfile
import json
import contextlib
import io
from pathlib import Path
from src.weave.graph_evidence import validate_graph_weave_delta, validate_graph_smoke
from src.weave.graph_evidence import validate_graph_delta


def snapshots():
    before=[];after=[]
    for rank in range(4):
        row=dict(rank=rank,graph_mode='full',backend='FullCudaGraphBackend',numerical_policy='record_only',source_sha256={'p':'a'*64},observer_source_sha256='b'*64,active=False,sequence=1,captured_batches=list(range(1,33)),capture_attempts=32,capture_successes=32,capture_failures=0,successful_replays=2,failed_replays=0,raw_batch_histogram={'32':2},captured_batch_histogram={'32':2},model_forwards={'capture':32,'warmup':32,'eager_decode':0,'prefill':1,'other':0})
        before.append(row);new=copy.deepcopy(row);new.update(sequence=2,successful_replays=5,raw_batch_histogram={'32':5},captured_batch_histogram={'32':5});after.append(new)
    return before,after


class GraphEvidenceTests(unittest.TestCase):
    def test_real_replays(self):
        a,b=snapshots();self.assertEqual(validate_graph_delta(a,b)['replays'],3)

    def test_reject_phony_padding_fallback_recapture_rank_drift(self):
        changes=[lambda r:r.update(captured_batches=[32]),lambda r:r['model_forwards'].update(eager_decode=1),lambda r:r.update(capture_attempts=33),lambda r:r.update(raw_batch_histogram={'31':3,'32':2}),lambda r:r.update(successful_replays=6),lambda r:r.update(source_sha256={'p':'c'*64})]
        for change in changes:
            a,b=snapshots();change(b[3])
            with self.subTest(change=change),self.assertRaises(ValueError):validate_graph_delta(a,b)

    def test_service_defaults_and_graph_switch(self):
        from scripts.run_c32_comparison import service_command
        args=('native','/w','/m','n',30001,30110,'a'*64)
        old=service_command(*args);self.assertEqual(old[old.index('--decode-graph')+1],'disabled');self.assertNotIn('--graph-c32',old)
        full=service_command(*args,decode_graph='full');self.assertIn('--graph-c32',full);self.assertEqual(full[full.index('--decode-graph')+1],'full')

    def test_full_preview_common_graph_contract_and_fixed_gpqa_workload(self):
        from scripts.run_gpqa_c32 import main
        output=io.StringIO()
        with patch('scripts.run_gpqa_c32.subprocess.run',side_effect=AssertionError('preview must stay offline')),contextlib.redirect_stdout(output):
            self.assertEqual(main(['--work-root','/tmp/work','--records','/tmp/records','--prefill-evidence-dir','/tmp/proof','--decode-graph','full']),0)
        report=json.loads(output.getvalue());self.assertEqual(report['seconds'],1200);self.assertEqual(report['concurrency'],32)
        self.assertEqual(report['max_new_tokens'],32768);self.assertFalse(report['ignore_eos']);self.assertTrue(report['graph_smoke_required'])
        for command in report['groups'].values():
            self.assertEqual(command[command.index('--decode-graph')+1],'full');self.assertIn('--graph-c32',command)
            self.assertEqual(command[command.index('--graph-evidence-port-base')+1],'30310')

    def test_baseline_uses_one_total_deadline(self):
        from scripts.run_gpqa_c32 import baseline_coverage
        with patch('scripts.run_gpqa_c32.time.monotonic',side_effect=[1.,2.,3.,4.,5.,6.,7.,8.]),patch('scripts.run_gpqa_c32.http_json',return_value={}) as http:
            self.assertEqual(len(baseline_coverage(30310,deadline=20.,endpoint='graph-coverage')),4)
            self.assertEqual([c.kwargs['timeout'] for c in http.call_args_list],[19.,17.,15.,13.])
            self.assertTrue(all('/graph-coverage' in c.args[0] for c in http.call_args_list))
        with patch('scripts.run_gpqa_c32.time.monotonic',return_value=21.):
            with self.assertRaises(TimeoutError):baseline_coverage(30310,deadline=20.)

    def test_graph_gate_existing_image_dynamic_allocation(self):
        from scripts.run_graph_gate import graph_gate_command
        cmd=graph_gate_command('sha256:abc','/w','/m','job','3,5,6,7')
        self.assertEqual(cmd[cmd.index('--gpus')+1], '\"device=3,5,6,7\"')
        self.assertIn('--pull=never',cmd);self.assertIn('/workspace/tests/gpu/check_hopper_graph.py',cmd)
        self.assertEqual(cmd[-1],'/work/job/graph-gate')
        self.assertNotIn('/workspace/tests/gpu/check_hopper_serving.py',cmd)

    def test_weave_actual_collectives_and_no_python_fallback(self):
        keys=('source_sha256','candidate_source_sha256','assembly_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256','graph_runtime_source_sha256','checkpoint_content_sha256')
        identity={k:'a'*64 for k in keys};before=[];after=[]
        calls={k:0 for k in ('hopper_decode','staged_decode','persistent_decode','persistent_prefill','persistent_other','hopper_prefill')}
        for rank in range(4):
            row={k:identity[k] for k in keys if k!='checkpoint_content_sha256'}
            row.update(rank=rank,graph_enabled=True,kernel='tiled',hot_cache=True,fixed_plan=[1,1,0],profile_bank_sha256=None,execution_profile_configuration=None,decode_implementation='hopper_fp8',prefill_implementation='hopper_fp8',numerical_policy='record_only',profile_checkpoint_content_sha256=identity['checkpoint_content_sha256'],model_batch_counts={'active':False,'failed_forwards':0,'decode_valid_batch_histogram':{}},device_slot_collectives=0,additional_prefill_slot_collectives=0,layers={str(i):{'device_calls':[0,0,0],'implementation_calls':dict(calls),'capture_calls':32,'python_decode_calls':0} for i in range(43)})
            before.append(row);new=copy.deepcopy(row);new.update(device_slot_collectives=129,additional_prefill_slot_collectives=43)
            new['model_batch_counts']['decode_valid_batch_histogram']={'32':3}
            for layer in new['layers'].values():layer['device_calls']=[1,3,0];layer['implementation_calls']['hopper_prefill']=1
            after.append(new)
        check={'replays':3,'decode_batch32_forwards_rank0':3,'ranks':{str(rank):{'decode_batch_histogram':{'32':3}} for rank in range(4)}}
        self.assertEqual(validate_graph_weave_delta(before,after,check,identity)['status'],'passed')
        for change in (lambda row:row.update(device_slot_collectives=128),lambda row:row['layers']['42']['implementation_calls'].update(hopper_decode=1),lambda row:row['layers']['42'].update(device_calls=[1,2,0]),lambda row:row.update(additional_prefill_slot_collectives=42)):
            wrong=copy.deepcopy(after);change(wrong[2])
            with self.assertRaises(ValueError):validate_graph_weave_delta(before,wrong,check,identity)

    def test_complete_smoke_and_stale_matrix_counter_rejection(self):
        from tests.gpu.check_hopper_graph import expected_cases
        identity={k:'a'*64 for k in ('source_sha256','candidate_source_sha256','assembly_source_sha256','graph_runtime_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256')};identity['image_id']='sha256:123'
        rows=[]
        hashes={k:v for k,v in identity.items() if k not in ('source_sha256','image_id')};hashes['bf16_source_sha256']=identity['source_sha256']
        for rank in range(4):
            row=dict(hashes,rank=rank,world_size=4,status='passed',numerical_policy='record_only',raw_content_exported=False,graph_capture_count=64,replay_count=38,replay_device_collective_delta=38,case_count=38,capture_batches=list(range(32,0,-1)),capture_handles=[0,1],captures=[{'batch':b,'handle':h} for h in (0,1) for b in range(32,0,-1)],capture_device_collective_delta=0,source_hashes_after=hashes,runtime={'image_id':identity['image_id'],'capability':[9,0],'device_count':4,'device_name':'NVIDIA H100'},device_slot_collectives=140,eager_slot_collectives=102,prefill_interleave={'batch':65,'finite':True,'additional_slot_collectives':1,'separate_workspace':True})
            row['cases']=[dict(case,finite=True,metadata_exact=True,assembly_exact=True,stable_addresses=True,binding_alive=True,device_collective_delta=1,device_counter_before=i,device_counter_after=i+1,input_update_verified=True,oracle_tokens=min(case['batch'],2),numerical={k:{'finite':True,'relative_l2':1e300,'max_abs':1e300} for k in ('eager_graph','fp8_oracle','bf16_oracle','bf16_reference')}) for i,case in enumerate(expected_cases())]
            rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            def write(values):
                for rank,row in enumerate(values):(Path(tmp)/f'hopper-graph-rank{rank}.json').write_text(json.dumps(row))
            write(rows);self.assertEqual(validate_graph_smoke(tmp,identity)['status'],'passed')
            changes=(lambda r:r['cases'].pop(),lambda r:r.update(graph_runtime_source_sha256='b'*64),lambda r:r.update(replay_device_collective_delta=37),lambda r:r['cases'][0].update(device_collective_delta=0),lambda r:r['cases'][0]['numerical']['fp8_oracle'].update(relative_l2=float('nan')))
            for change in changes:
                wrong=copy.deepcopy(rows);change(wrong[2]);write(wrong)
                with self.assertRaises(ValueError):validate_graph_smoke(tmp,identity)
            for name in ('eager_graph','fp8_oracle','bf16_oracle','bf16_reference'):
                for field in ('relative_l2','max_abs'):
                    for invalid in ('missing',None,'1.0',True,-1.,float('inf'),float('-inf'),float('nan')):
                        wrong=copy.deepcopy(rows);metric=wrong[2]['cases'][0]['numerical'][name]
                        if invalid=='missing':metric.pop(field)
                        else:metric[field]=invalid
                        write(wrong)
                        with self.subTest(oracle=name,field=field,invalid=invalid),self.assertRaises(ValueError):validate_graph_smoke(tmp,identity)
            wrong=copy.deepcopy(rows)
            for case in wrong[2]['cases']:
                case['numerical']={key:{'finite':True} for key in case['numerical']}
            write(wrong)
            with self.assertRaises(ValueError):validate_graph_smoke(tmp,identity)

    def test_missing_smoke_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):validate_graph_smoke(tmp,{})


if __name__=='__main__':unittest.main()
