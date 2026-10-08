import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.weave import calibration as cal
from tests.cpu.test_calibration import IDENTITY,point
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
CASE={'tokens':1024,'c':131,'K':8,'mode':'mixed'}


class CalibrationDiagnosticsTest(unittest.TestCase):
    def test_counter_fault_retains_origin_before_collective_string_conversion(self):
        capture=getattr(cal,'capture_local_action',None)
        self.assertTrue(callable(capture),'preserve original GPU errors before rank agreement')
        with tempfile.TemporaryDirectory() as tmp:
            journal=cal.CalibrationJournal(Path(tmp),IDENTITY,'counter-fault',100.,1200.,clock=lambda:102.)
            journal.begin(CASE,1,elapsed_seconds=2.,completed_cases=285)
            journal.stage('workload_counter_read')
            def counter_read_origin(): raise RuntimeError('counter transfer failure')
            value,error=capture(counter_read_origin,journal=journal)
            self.assertIsNone(value)
            self.assertIn('counter transfer failure',error)
            failure=json.loads((Path(tmp)/'failure-rank0-counter-fault.json').read_text())
            self.assertIn('counter_read_origin',failure['traceback'])
            self.assertEqual(failure['gpu_stage'],'workload_counter_read')
            journal.fail(RuntimeError('collective wrapper'))
            self.assertEqual(json.loads(journal.failure_path.read_text()),failure)

    def test_host_deadline_stops_only_its_owned_container_and_returns124(self):
        from src.weave.container_lifecycle import run_container
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bindir=root/'bin';bindir.mkdir();live=root/'live'
            docker=bindir/'docker'
            docker.write_text('#!'+sys.executable+'\nimport os,sys,time\nfrom pathlib import Path\n'
                'live=Path(os.environ["FAKE_LIVE"])\n'
                'if sys.argv[1]=="run":\n'
                '    live.write_text("owned")\n'
                '    while live.exists(): time.sleep(.02)\n'
                'elif sys.argv[1] in ("stop","rm"): live.unlink(missing_ok=True)\n'
                'else: print("No such object",file=sys.stderr);sys.exit(1)\n')
            docker.chmod(0o755)
            with patch.dict(os.environ,{'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
                                        'FAKE_LIVE':str(live)}):
                try:
                    result=run_container(['docker','run','--rm','--name','weave-deadline-test'],
                        'weave-deadline-test',root/'lifecycle.json',expired=live.exists)
                except TypeError:
                    self.fail('calibration container owner must enforce a measuring deadline')
            self.assertEqual(result['exit_code'],124)
            self.assertEqual(result['cleanup_status'],'absent')
            self.assertTrue(result['deadline_expired'])
            self.assertFalse(live.exists())

    def test_near_deadline_does_not_shrink_device_guard_or_commit_late_point(self):
        execute=getattr(cal,'execute_gpu_iteration',None)
        self.assertTrue(callable(execute),'GPU guard must be separate from measurement remaining time')
        with tempfile.TemporaryDirectory() as tmp:
            store=cal.TimingStore(Path(tmp)/'timings-rank0.json',IDENTITY)
            now=[1199.99];calls=[]
            def launch(seconds):
                # This healthy operation takes100ms; a10ms device guard would trap.
                if seconds<.1: raise RuntimeError('CUDA error: unspecified launch failure')
                calls.append(seconds);now[0]+=.1
            def measure(case):
                execute(launch,lambda:None,kernel_timeout_seconds=60.)
                return point()
            result=cal.run_calibration(store,measure,lambda:now[0]>=1200.,lambda:now[0],1200.)
            self.assertEqual(result['status'],'partial')
            self.assertEqual(calls,[60.])
            self.assertEqual(store.points,[])
            self.assertEqual(len(store.excluded_points),1)

    def test_sync_fault_records_exact_case_iteration_stage_and_traceback(self):
        journal_type=getattr(cal,'CalibrationJournal',None)
        self.assertTrue(callable(journal_type),'persist the active GPU case before launch')
        with tempfile.TemporaryDirectory() as tmp:
            journal=journal_type(Path(tmp),IDENTITY,'attempt-new',100.,1200.)
            journal.begin(CASE,2,elapsed_seconds=41.,completed_cases=285)
            def sync(): raise RuntimeError('CUDA error: unspecified launch failure')
            with self.assertRaisesRegex(RuntimeError,'launch failure'):
                cal.execute_gpu_iteration(lambda seconds:None,sync,journal=journal)
            failure=json.loads((Path(tmp)/'failure-rank0-attempt-new.json').read_text())
            self.assertEqual(failure['active_case'],CASE)
            self.assertEqual(failure['iteration'],2)
            self.assertEqual(failure['gpu_stage'],'cuda_synchronize')
            self.assertEqual(failure['completed_cases'],285)
            self.assertEqual(failure['status'],'failed')
            self.assertIn('RuntimeError',failure['traceback'])
            marker=json.loads((Path(tmp)/'progress-rank0.json').read_text())
            self.assertEqual(marker['measuring_started_monotonic'],100.)
            self.assertEqual(marker['invocation_id'],'attempt-new')

    def test_diagnostic_run_never_changes_existing_points_or_publishes_profile(self):
        diagnose=getattr(cal,'run_diagnostic_case',None)
        self.assertTrue(callable(diagnose),'diagnostic samples must be separate from formal calibration')
        with tempfile.TemporaryDirectory() as tmp:
            store=cal.TimingStore(Path(tmp)/'timings-rank0.json',IDENTITY)
            store.append(point(),elapsed_seconds=1.,budget_seconds=1200.)
            original=store.path.read_bytes()
            result=diagnose(store,CASE,lambda case:point(1024,131,8,'mixed'),lambda:False)
            self.assertEqual(result['status'],'diagnostic_passed')
            self.assertFalse(result['used_for_calibration'])
            self.assertEqual(store.path.read_bytes(),original)
            self.assertFalse(list(Path(tmp).glob('profile-bank*')))

    def test_failed_calibration_export_contains_active_configuration(self):
        from src.weave.experiment_export import export_run
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir();directory=source/'cal';directory.mkdir()
            (source/'experiment.json').write_text(json.dumps({'status':'failed','performance':'not_run',
                'calibration_dir':str(directory)}))
            failure={'status':'failed','rank':0,'invocation_id':'attempt-new','active_case':CASE}
            (directory/'failure-rank0-attempt-new.json').write_text(json.dumps(failure))
            dest=Path(tmp)/'public';export_run(source,dest)
            file=dest/'calibration/failure-rank0-attempt-new.json'
            self.assertTrue(file.is_file(),'failed runs need actionable per-case evidence')
            self.assertEqual(json.loads(file.read_text()),failure)

    def test_archive_failed_experiment_preserves_failure_status(self):
        from scripts import run_full_experiment as full
        archive=getattr(full,'archive_evidence',None)
        self.assertTrue(callable(archive),'archive failures before rethrowing the original error')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir();dest=Path(tmp)/'public'
            report={'status':'failed','performance':'not_run','error':'original CUDA fault'}
            archive(report,source,dest)
            public=json.loads((dest/'experiment.json').read_text())
            self.assertEqual(public['status'],'failed')
            self.assertEqual(public['error'],'original CUDA fault')

    def test_diagnostic_docker_enables_blocking_and_retains_offline_gpu_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();bindir=root/'bin';bindir.mkdir();capture=root/'argv.json'
            docker=bindir/'docker'
            docker.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                'if sys.argv[1:3]==["image","inspect"]: print("sha256:"+"a"*64)\n'
                'elif sys.argv[1]=="run":\n'
                '    open(os.environ["DOCKER_CAPTURE"],"w").write(json.dumps(sys.argv[1:]));sys.exit(1)\n'
                'else: print("No such object",file=sys.stderr);sys.exit(1)\n')
            docker.chmod(0o755)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7','DOCKER_CAPTURE':str(capture),
                 'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            command=[sys.executable,str(ROOT/'scripts/run_calibration_job.py'),'--model-path',str(root/'model'),
                '--work-root',str(root),'--core-run-dir',str(root/'core'),
                '--output-dir',str(root/'diagnostic'),'--lifecycle-path',str(root/'lifecycle.json'),
                '--diagnose-case','1024,131,8,mixed','--export-dir',str(root/'public')]
            run=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,1,run.stderr)
            self.assertTrue(capture.exists(),run.stderr)
            args=json.loads(capture.read_text())
            self.assertIn('CUDA_LAUNCH_BLOCKING=1',args)
            self.assertIn('--pull=never',args)
            self.assertIn('HF_HUB_OFFLINE=1',args)
            self.assertEqual(args[args.index('--gpus')+1],'"device=4,5,6,7"')
            self.assertEqual(args[args.index('--diagnose-case')+1],'1024,131,8,mixed')
            self.assertEqual(json.loads((root/'lifecycle.json').read_text())['cleanup_status'],'absent')
            self.assertEqual(json.loads((root/'public/manifest.json').read_text())['experiment_status'],'failed')
