import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    from src.weave.execution_profile import TraceConfig,summarize_profile
except ImportError:
    TraceConfig=summarize_profile=None

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    identity={'rank':0,'source_sha256':'a'*64,'profile_bank_sha256':'b'*64,
              'checkpoint_content_sha256':'c'*64,'kernel':'tiled','graph_enabled':True,
              'device_key':'NVIDIA H100 80GB HBM3;sm90;132'}
    heads=[[0]*18 for _ in range(2)]
    totals=[[[0]*32 for _ in range(2)] for _ in range(2)]
    records=[[[[0]*32 for _ in range(3)] for _ in range(2)] for _ in range(2)]
    heads[0][0:2]=[5,3];heads[0][10]=3  # c=1,K=1
    for worker in range(2):
        for field,value in ((12,60 if worker==0 else 0),(14,240),(17,90 if worker==0 else 0),
                            (18,0 if worker==0 else 90),(20,90),(22,90),(23,30),
                            (24,3),(25,3 if worker==0 else 0),(26,3 if worker==0 else 0),
                            (27,3 if worker==0 else 0),(28,600),(29,30 if worker==0 else 0),(30,3)):
            totals[0][worker][field]=value
    for slot,seq in [(0,4),(1,2)]:
        meta=records[0][slot][2]
        meta[:14]=[seq,99,1,1,1,1,1,0,1,1,1,1,2,0]
        for worker in range(2):
            row=records[0][slot][worker]
            offset=seq*100+100+worker*10
            row[:12]=[offset,offset+10,offset+10,offset+30,
                      offset+30,offset+110,offset+110,offset+140,
                      offset+140,offset+170,offset+170,offset+200]
            row[12]=20;row[14]=80;row[17]=30;row[20]=30
            row[22]=30;row[23]=10;row[24]=1;row[26]=1;row[27]=1
            row[28]=200;row[30]=1
            if worker==0: row[25]=1;row[29]=10
            else:
                row[2]=row[3]=row[12]=row[17]=row[26]=row[27]=0
                row[18]=30
    return identity,heads,totals,records


class ExecutionProfileTest(unittest.TestCase):
    def test_fixed_eager_report_binds_plan_without_inventing_a_bank(self):
        identity,heads,totals,records=fixture()
        identity.update(graph_enabled=False,profile_bank_sha256=None,
                        schedule_mode='explicit_fixed',fixed_plan=[1,1,1])
        result=summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,
                                 heads,totals,records,workers=2)
        self.assertEqual(result['fixed_plan'],[1,1,1])
        self.assertIsNone(result['profile_bank_sha256'])
        self.assertFalse(result['graph_enabled'])
        self.assertFalse(result['used_for_formal_performance'])
        identity['fixed_plan']=[0,1,1]
        with self.assertRaises(ValueError):
            summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,
                              heads,totals,records,workers=2)

    def test_fixed_profile_preview_requires_explicit_eager_and_checkpoint(self):
        base=[sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--backend','weave_functional',
              '--kernel','tiled','--execution-profile','--execution-profile-fixed-plan',
              '--checkpoint-content-sha256','a'*64]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6',
             'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'}
        good=subprocess.run(base+['--decode-graph','disabled'],env=env,capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        argv=json.loads(good.stdout)['docker_argv']
        self.assertIn('WEAVE_FORWARD_PROFILE=1',argv)
        self.assertIn('WEAVE_FIXED_PLAN=1,1,0',argv)
        for suffix in (['--decode-graph','full'],['--profile','/tmp/profile.json']):
            bad=subprocess.run(base+suffix,env=env,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)

    def test_overlapping_worker_time_is_not_summed_as_layer_latency(self):
        self.assertIsNotNone(summarize_profile)
        identity,heads,totals,records=fixture()
        result=summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,
                                 heads,totals,records,workers=2)
        self.assertFalse(result['used_for_formal_performance'])
        value=result['layers']['0']['prefill']
        self.assertEqual(value['calls'],5)
        self.assertEqual(value['sampled_invocations'],3)
        self.assertEqual(value['overwritten_samples'],1)
        self.assertEqual(value['worker_totals_ns']['compute_work'],480)
        self.assertEqual(value['plan_histogram'],[{'c':1,'K':1,'sampled_count':3}])
        recent=value['recent_samples']
        self.assertEqual([v['sequence'] for v in recent],[2,4])
        self.assertEqual(recent[0]['kernel_span_ns'],210)
        self.assertEqual(recent[0]['phase_envelopes_ns']['compute'],90)
        self.assertEqual(recent[0].get('phase_intervals_ns',{}).get('compute'),[30,120])
        self.assertEqual(recent[0]['worker_totals_ns']['compute_work'],160)

    def test_corrupt_ring_or_counter_cannot_be_reported_as_profiled(self):
        self.assertIsNotNone(summarize_profile)
        for change in ('counter','timer','slot','plan'):
            identity,heads,totals,records=fixture()
            if change=='counter': heads[0][1]=4
            if change=='timer': records[0][0][0][11]=1
            if change=='slot': records[0][0][2][0]=3
            if change=='plan': records[0][0][2][4]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,
                                  heads,totals,records,workers=2)

    def test_configuration_is_bounded_before_gpu_allocation(self):
        self.assertIsNotNone(TraceConfig)
        for options in ({'stride':0},{'capacity':0},{'capacity':65},{'layers':0}):
            with self.subTest(options=options),self.assertRaises(ValueError): TraceConfig(**options)

    def test_impossible_phase_order_and_work_counts_are_rejected(self):
        for corruption in ('phase_order','missing_stage','stealing','tiles','returns','rows','totals'):
            identity,heads,totals,records=fixture()
            if corruption=='phase_order': records[0][0][0][1]=records[0][0][0][9]+1
            if corruption=='missing_stage': records[0][0][0][4]=0
            if corruption=='stealing': records[0][0][0][25]=999
            if corruption=='tiles': records[0][0][2][12]+=1
            if corruption=='returns': records[0][0][2][10]+=1
            if corruption=='rows': records[0][0][2][2]+=1
            if corruption=='totals': totals[0][0][14]=0
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):
                summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,
                                  heads,totals,records,workers=2)

    def test_diagnostic_caller_cannot_override_safety_labels(self):
        identity,heads,totals,records=fixture();identity['used_for_formal_performance']=True
        with self.assertRaises(ValueError):
            summarize_profile(TraceConfig(stride=2,capacity=2,layers=1),identity,heads,totals,records,workers=2)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_cuda_shared_sampling_and_totals_helpers_on_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'execution_trace'
            build=subprocess.run(['c++','-std=c++17','-I',str(ROOT/'csrc'),
                str(ROOT/'tests/cpp/test_execution_trace.cpp'),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_native_launch_rejects_weave_instrumentation(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_native_ep.py'),
            '--execution-profile'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6'},
            capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires weave_functional',result.stderr)

    def test_weave_preview_passes_bounded_profile_configuration(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_native_ep.py'),
            '--backend','weave_functional','--kernel','tiled','--execution-profile',
            '--execution-profile-stride','64','--execution-profile-capacity','4'],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1,3,5,7'},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads(result.stdout)['docker_argv']
        self.assertIn('WEAVE_EXECUTION_PROFILE=1',argv)
        self.assertIn('WEAVE_EXECUTION_PROFILE_STRIDE=64',argv)
        self.assertIn('WEAVE_EXECUTION_PROFILE_CAPACITY=4',argv)

    def test_profile_artifacts_are_explicitly_exported_with_hashes(self):
        from src.weave.experiment_export import export_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'weave').mkdir(parents=True)
            (run/'experiment.json').write_text(json.dumps({'status':'budgeted_diagnostic_complete',
                'performance':'instrumented_diagnostic'}))
            for rank in range(4):
                (run/'weave'/f'execution-profile-rank{rank}.json').write_text(json.dumps({
                    'rank':rank,'status':'profiled','used_for_formal_performance':False}))
            (run/'weave'/'private.json').write_text(json.dumps({'output_ids':[123]}))
            out=root/'public';manifest=export_run(run,out)
            self.assertEqual(len(manifest['files']),5)
            self.assertTrue((out/'weave/execution-profile-rank3.json').is_file())
            self.assertFalse((out/'weave/private.json').exists())

    def test_fixed_diagnostic_export_keeps_launch_and_summary_but_excludes_raw_logs(self):
        from src.weave.experiment_export import export_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'weave').mkdir(parents=True)
            (run/'experiment.json').write_text(json.dumps({'status':'failed','scope':'c32_fixed_eager_diagnostic'}))
            (run/'launch.json').write_text(json.dumps({'docker_argv':['--fixed-plan','1,1,0']}))
            (run/'weave/diagnostic-summary.json').write_text(json.dumps({'model_decode_rows_rank0':32}))
            (run/'benchmark-private.log').write_text('private prompt text')
            (run/'bench-private.jsonl').write_text(json.dumps({'text':'private output'}))
            manifest=export_run(run,root/'public')
            self.assertEqual(set(manifest['files']),{'experiment.json','launch.json','weave/diagnostic-summary.json'})

    def test_diagnostic_mode_is_recorded_even_when_setup_fails(self):
        from tests.cpu.test_gpqa_performance import make_records
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();records=make_records(root);bindir=root/'bin';bindir.mkdir()
            tool=bindir/'docker';tool.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(7)\n');tool.chmod(0o755)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_full_experiment.py'),
                '--work-root',str(root/'work'),'--records',str(records),'--execution-profile'],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6','PATH':str(bindir)+os.pathsep+os.environ['PATH']},
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            files=list((root/'work').glob('full-experiment-*/experiment.json'))
            self.assertEqual(len(files),1,result.stderr)
            report=json.loads(files[0].read_text())
            self.assertEqual(report['measurement_mode'],'instrumented_diagnostic')
            self.assertFalse(report['used_for_formal_performance'])
