import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.run_c32_comparison import service_command,client_command,gate_command,export_public

ROOT=Path(__file__).resolve().parents[2]
class StallLaunchTest(unittest.TestCase):
    def test_all_groups_get_shared_diagnostic_hook(self):
        for variant in ('native','staged_direct','hopper_fp8'):
            cmd=service_command(variant,Path('/work'),Path('/model'),'native-ep-test',30001,30110,'a'*64,stall_diagnostic=True,stall_port_base=30210)
            cmd.remove('--execute')
            run=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1,3,5,7'},capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            docker=json.loads(run.stdout)['docker_argv']
            self.assertIn('WEAVE_STALL_DIAGNOSTIC=1',docker)
            self.assertIn('WEAVE_STALL_PORT_BASE=30210',docker)
            self.assertIn('PYTHONPATH=/workspace/timing_hooks:/workspace/src',docker)
            self.assertEqual(docker[docker.index('--tp-size')+1],'4')
    def test_acceptance_never_arms_and_measurement_does(self):
        for acceptance in (True,False):
            cmd=client_command('image',Path('/work'),Path('/data'),'out','name',30001,600,acceptance,stall_diagnostic=True,stall_port_base=30210)
            self.assertEqual('--stall-port-base' in cmd,not acceptance)
    def test_gate_chains_smoke_before_assembly_without_shell(self):
        cmd=gate_command('image',Path('/work'),Path('/model'),'name','1,3,5,7','hopper',stall_diagnostic=True)
        self.assertIn('/workspace/scripts/run_stall_gate.py',cmd)
        self.assertNotIn('sh',cmd)
    def test_native_rejects_graph_and_colliding_ports(self):
        for extra in (['--decode-graph','full'],['--decode-graph','disabled','--stall-port-base','30112']):
            run=subprocess.run([sys.executable,'scripts/run_native_ep.py','--stall-diagnostic','--request-timing']+extra,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3'},capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
    def test_export_diagnostic_allowlist(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';(run/'native').mkdir(parents=True);(run/'online-gate').mkdir()
            (run/'experiment.json').write_text(json.dumps({'status':'failed'}))
            names=['native/client-stall.json','native/stall-analysis.json','native/stall-rank0.json','online-gate/stall-smoke-rank0.json']
            for name in names:(run/name).write_text('{}')
            (run/'native/stall-private.log').write_text('private')
            manifest=export_public(run,root/'public')
            for name in names:self.assertIn(name,manifest['files'])
            self.assertNotIn('native/stall-private.log',manifest['files'])
    def test_convenience_wrapper_cannot_select_staged(self):
        from unittest.mock import patch
        from scripts.run_prefill_stall_diagnostic import main
        with patch('scripts.run_prefill_stall_diagnostic.comparison_main',return_value=0) as run:
            self.assertEqual(main(['--comparison-suite','staged']),0)
        self.assertEqual(run.call_args.args[0][-3:],['--comparison-suite','hopper','--stall-diagnostic'])
    def test_snapshot_reads_four_ranks_then_analyzes_persisted_client_and_window(self):
        from unittest.mock import patch
        from scripts.run_c32_comparison import snapshot_stalls
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ('client-stall.json','load-window.json'):(folder/name).write_text('{}')
            calls=[]
            def read(url,timeout):
                calls.append(url);return {'rank':len(calls)-1}
            with patch('scripts.run_c32_comparison.http_json',side_effect=read),patch('src.weave.stall_analysis.analyze_stalls',return_value={'status':'passed'}) as analyze:
                snapshot_stalls(folder,30210)
            self.assertEqual(calls,[f'http://127.0.0.1:{30210+r}/snapshot' for r in range(4)])
            self.assertEqual(len(analyze.call_args.args[0]),4)
            self.assertFalse(json.loads((folder/'stall-analysis.json').read_text())['used_for_formal_performance'])
    def test_stall_gate_runs_sequential_argv_and_stops_on_smoke_failure(self):
        from unittest.mock import patch
        from scripts.run_stall_gate import main
        with patch('scripts.run_stall_gate.subprocess.run') as run:
            main(['--output-dir','/work/gate'])
        self.assertEqual(run.call_count,2)
        self.assertIn('/workspace/tests/gpu/check_stall_timeline.py',run.call_args_list[0].args[0])
        self.assertIn('/workspace/tests/gpu/check_hopper_serving.py',run.call_args_list[1].args[0])
        with patch('scripts.run_stall_gate.subprocess.run',side_effect=subprocess.CalledProcessError(1,['smoke'])) as run:
            with self.assertRaises(subprocess.CalledProcessError):main(['--output-dir','/work/gate'])
        self.assertEqual(run.call_count,1)
    def test_staged_suite_rejects_opt_in_before_gpu_or_filesystem(self):
        from scripts.run_c32_comparison import main
        with self.assertRaises(SystemExit):main(['--work-root','/tmp/work','--dataset','/tmp/data','--stall-diagnostic'])
    def test_snapshot_rejects_wrong_expected_source_or_group(self):
        from unittest.mock import patch
        from scripts.run_c32_comparison import snapshot_stalls
        with tempfile.TemporaryDirectory() as temp,patch('scripts.run_c32_comparison.http_json',return_value={'rank':0,'source_sha256':'bad','identity':{'image':'image','candidate':'native'}}):
            with self.assertRaisesRegex(ValueError,'stall_source_identity_differs'):
                snapshot_stalls(Path(temp),30210,expected_source='a'*64,expected_image='image',expected_candidate='native')
    def test_smoke_validation_requires_all_four_successful_reports(self):
        from scripts.run_c32_comparison import validate_stall_smoke
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for rank in range(4):(folder/f'stall-smoke-rank{rank}.json').write_text(json.dumps({'rank':rank,'status':'passed'}))
            validate_stall_smoke(folder)
            (folder/'stall-smoke-rank3.json').write_text(json.dumps({'rank':3,'status':'failed'}))
            with self.assertRaisesRegex(ValueError,'stall_smoke_incomplete'):validate_stall_smoke(folder)
