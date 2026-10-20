import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.run_gpqa_c32 import main, export_public
from scripts.run_graph_gate import native_p0_gate_command
from tests.cpu.test_native_p0_trace import event
from src.weave.native_p0_trace import parse_chrome_trace
from src.weave.native_p0 import derive_layer_metrics
from src.weave.native_p0_evidence import native_checkpoint_proof, summarize_native_p0, validate_native_sample, validate_native_p0_smoke


def sample(sequence, ordinal, scale=1.):
    layers={}
    for layer in range(43):
        base=layer*2.*scale
        intervals={}
        for name,(start,end) in {'moe':(0,2),'routed':(.1,1.4),'shared':(.2,1.6),
             'gate_up':(.2,.6),'activation':(.6,.7),'down':(.7,1.2),'grouping':(.1,.2),
             'shared_merge':(1.6,1.7),'all_reduce':(1.7,2)}.items():
            intervals[name]=dict(start_ms=base+start*scale,end_ms=base+end*scale,duration_ms=(end-start)*scale)
        layers[str(layer)]=dict(intervals=intervals,**derive_layer_metrics(intervals),grouping=dict(
            valid_local_pairs=3,valid_groups=2,active_experts=2,max_rows=2,row_histogram={'1':1,'2':1},
            num_tokens_post_pad=16,block_size=8,valid_pair_capacity=256))
    return dict(sequence=sequence,batch_size=32,b32_ordinal=ordinal,whole_graph_ms=90*scale,layers=layers,launch_trace_active=False)


def snapshots():
    before=[];after=[]
    for rank in range(4):
        p=dict(kind='native_p0',rank=rank,profile_source_sha256='a'*64,measurement_mode='instrumented_diagnostic',
               numerical_policy='record_only',external_events=True,configuration=dict(stride=2,capacity=2,layers=43,captured_batch_size=32),
               b32_seen=0,sampled_total=0,aborted_samples=0,samples=[],
               layer_metadata={str(i):dict(captured_path='forward_normal_dual_stream',marlin_calls=[dict(role=r,size_m=32,size_n=2048,size_k=4096,moe_block_size=8,top_k=8,is_ep=True,bias_present=False,is_zp_float=False,b_q_type_id=5) for r in ('gate_up','down')]) for i in range(43)},
               launch_trace=dict(parse_chrome_trace([event(ts=1000+i*20) for i in range(86)]),rank=rank))
        before.append(dict(rank=rank,successful_replays=1,raw_batch_histogram={},stage_profile=deepcopy(p)))
        p.update(b32_seen=4,sampled_total=2,samples=[sample(2,2,1+rank*.1),sample(4,4,1+rank*.1)])
        after.append(dict(rank=rank,successful_replays=4,raw_batch_histogram={'32':4},stage_profile=p))
    return before,after


class NativeP0EvidenceTests(unittest.TestCase):
    def test_overlap_and_same_critical_rank(self):
        before,after=snapshots()
        result=summarize_native_p0(before,after,profile_source='a'*64,stride=2,capacity=2)
        self.assertEqual(result['critical_rank_histogram'],{'3':2})
        self.assertAlmostEqual(result['critical_rank']['shared_tail_ms']['mean'],43*.2*1.3)
        self.assertAlmostEqual(result['critical_rank']['outside_moe_ms']['mean'],4*1.3)
        self.assertFalse(result['formal_performance'])
        layer=result['per_rank']['0']['layers']['0']
        self.assertAlmostEqual(layer['derived_ms']['shared_tail_ms']['mean'],.2)
        self.assertAlmostEqual(layer['intervals_relative_to_graph_ms']['shared']['start_ms']['mean'],.2)

    def test_optional_span_is_absent_not_zero(self):
        row=sample(2,2)
        for layer in row['layers'].values():
            del layer['intervals']['all_reduce']
        self.assertNotIn('all_reduce_ms',validate_native_sample(row))

    def test_reject_bad_freshness_coverage_and_histogram(self):
        for mutate in (
            lambda a:a[0]['stage_profile']['samples'][0].update(b32_ordinal=1),
            lambda a:a[0]['stage_profile']['samples'][0]['layers'].pop('42'),
            lambda a:a[0]['stage_profile']['samples'][0]['layers']['0']['grouping'].update(row_histogram={'1':9}),
            lambda a:a[0]['stage_profile']['launch_trace'].update(marlin_kernel_count=85),
            lambda a:a[0]['stage_profile']['launch_trace'].update(kernels=[]),
            lambda a:a[0]['stage_profile']['layer_metadata']['0']['marlin_calls'][0].update(moe_block_size=16),
            lambda a:a[0]['stage_profile'].update(profile_source_sha256='b'*64),
            lambda a:a[0]['stage_profile']['layer_metadata']['0'].update(captured_path=None),
        ):
            before,after=snapshots();mutate(after)
            with self.assertRaises(ValueError):summarize_native_p0(before,after,profile_source='a'*64,stride=2,capacity=2)

    def test_exclude_cupti_sample(self):
        before,after=snapshots()
        for row in after:row['stage_profile']['samples'][0]['launch_trace_active']=True
        result=summarize_native_p0(before,after,profile_source='a'*64,stride=2,capacity=2)
        self.assertEqual(result['sequences'],[4])

    def test_native_preview_no_candidate_proof_or_gpu_calls(self):
        output=io.StringIO()
        with patch('scripts.run_gpqa_c32.subprocess.run',side_effect=AssertionError('preview executed')),contextlib.redirect_stdout(output):
            self.assertEqual(main(['--native-p0','--decode-graph','full','--work-root','/work','--records','/records.jsonl']),0)
        value=json.loads(output.getvalue())
        self.assertEqual(set(value['groups']),{'native'})
        self.assertTrue(value['native_p0_gate_required']);self.assertFalse(value['graph_gate_required']);self.assertFalse(value['formal_gpqa'])
        command=value['groups']['native']
        self.assertIn('--native-p0',command)
        self.assertNotIn('--fixed-plan',command)
        self.assertNotIn('--hot-cache',command)
        self.assertEqual(command[command.index('--backend')+1],'native')
        self.assertEqual(value['seconds'],1200)

    def test_native_smoke_replaces_all_output_paths(self):
        cmd=native_p0_gate_command('sha256:existing',Path('/work'),Path('/model'),'job','0,2,4,6')
        self.assertIn('/workspace/tests/gpu/check_native_p0_events.py',cmd)
        self.assertTrue(all(cmd[i+1]=='/work/job/native-p0-gate' for i,v in enumerate(cmd) if v=='--output-dir'))
        self.assertIn('--pull=never',cmd)
        self.assertNotIn('/workspace/tests/gpu/check_hopper_graph.py',cmd)

    def test_export_no_raw_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'run';run.mkdir();(run/'native-p0-gate').mkdir()
            (run/'experiment.json').write_text('{"status":"diagnostic_complete"}')
            (run/'native-p0-summary.json').write_text('{"status":"diagnostic_complete"}')
            (run/'native-p0-gate/raw-trace.json').write_text('{"prompt":"private"}')
            result=export_public(run,root/'public')
            self.assertIn('native-p0-summary.json',result['files'])
            self.assertNotIn('native-p0-gate/raw-trace.json',result['files'])

    def test_smoke_requires_actual_sample_and_cleanup_evidence(self):
        from tests.gpu.check_native_p0_events import hashes
        from src.weave.graph_cleanup import CLEANUP_PHASES
        source=hashes();identity=dict(native_p0_source_sha256=source['profile_source_sha256'],image_id='sha256:existing')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for rank in range(4):
                cases=[]
                for seq in range(1,5):
                    item=sample(seq,seq);item['layers']={'0':item['layers']['0']}
                    cases.append(dict(sequence=seq,batch=32,sample=item,finite=True,fresh_events_verified=True,join_order_verified=True))
                profile=dict(rank=rank,profile_source_sha256=source['profile_source_sha256'],configuration=dict(stride=1,capacity=4,layers=1,captured_batch_size=32),
                    sampled_total=4,b32_seen=4,aborted_samples=0,samples=[c['sample'] for c in cases],
                    launch_trace=dict(parse_chrome_trace([event()],expected_layers=None),rank=rank))
                row=dict(status='passed',gpu_checks_status='passed',rank=rank,world_size=4,image_id=identity['image_id'],case_count=4,
                    replay_count=4,graph_capture_count=1,external_events=True,two_streams=True,nccl_all_reduce=True,
                    generic_launch_metadata_verified=True,numerical_policy='record_only',model_execution_evidence=False,**source,
                    source_hashes_after=source,cleanup=dict(status='complete',graphs_reset=1,completed_phases=list(CLEANUP_PHASES)),
                    runtime=dict(image_id=identity['image_id'],capability=[9,0],device_count=4,device_name='NVIDIA H100'),cases=cases,native_p0_profile=profile)
                (folder/f'native-p0-event-rank{rank}.json').write_text(json.dumps(row))
            self.assertEqual(validate_native_p0_smoke(folder,identity)['rank_count'],4)
            path=folder/'native-p0-event-rank0.json';row=json.loads(path.read_text());row['native_p0_profile']['samples'][0]['sequence']=99;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):validate_native_p0_smoke(folder,identity)

    def test_baseline_attestation_reuse_and_rehash(self):
        from src.weave.checkpoint_identity import checkpoint_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();baseline=root/'baseline';baseline.mkdir()
            (model/'config.json').write_text('{}');(model/'model.safetensors.index.json').write_text('{"weight_map":{"x":"shard"}}');(model/'shard').write_bytes(b'weights')
            identity=checkpoint_identity(model);p=baseline/'prefill-proof/core/checkpoint_identity.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(identity))
            (baseline/'manifest.json').write_text(json.dumps({'files':{p.relative_to(baseline).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()}}))
            with patch('src.weave.native_p0_evidence.checkpoint_identity',side_effect=AssertionError('unnecessary rehash')):
                result=native_checkpoint_proof(baseline,model)
            self.assertEqual(result['checkpoint_content_sha256'],identity['sha256'])
            (model/'shard').touch()
            self.assertEqual(native_checkpoint_proof(baseline,model)['checkpoint_verification'],'rehash_all_local_checkpoint_files')
            (model/'shard').write_bytes(b'other')
            with self.assertRaises(ValueError):native_checkpoint_proof(baseline,model)


if __name__=='__main__':unittest.main()
