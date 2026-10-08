import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.weave import calibration as cal,experiment as exp
from scripts import run_full_experiment as full
from tests.cpu.test_calibration import IDENTITY

ROOT=Path(__file__).resolve().parents[2]
CASE={'tokens':2048,'c':131,'K':1,'mode':'compute'}


class RecoveryControlsTest(unittest.TestCase):
    def test_export_root_rejects_private_run_alias_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp).resolve()/'work';work.mkdir()
            alias=Path(tmp)/'alias';alias.symlink_to(work,target_is_directory=True)
            for export_root in (work,alias,work/'full-experiment-test'):
                with self.subTest(export_root=export_root),self.assertRaisesRegex(ValueError,'overlap'):
                    exp.prepare_export_directory(work,'full-experiment-test',export_root)
            self.assertFalse((work/'full-experiment-test').exists())

    def test_successful_diagnostic_publishes_each_current_case_iteration_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=[120.]
            journal=cal.CalibrationJournal(Path(tmp),IDENTITY,'passing-case',100.,1200.,clock=lambda:now[0])
            publish=getattr(journal,'record_completed_iteration',None)
            self.assertTrue(callable(publish),'successful diagnostics need durable wall timings too')
            for iteration,duration in enumerate((5.,7.)):
                journal.begin(CASE,iteration,elapsed_seconds=now[0]-100.,completed_cases=0)
                def launch(seconds): now[0]+=duration
                cal.execute_gpu_iteration(launch,lambda:None,journal=journal)
                publish()
            value=json.loads(journal.path.read_text())
            self.assertEqual([v['gpu_iteration_elapsed_seconds'] for v in value['iterations']],[5.,7.])
            self.assertEqual(value['gpu_iteration_elapsed_seconds'],7.)
            self.assertEqual(value['gpu_iteration_finished_monotonic'],132.)

    def test_gpu_snapshot_timeout_can_be_archived_without_masking_kernel_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            error=subprocess.TimeoutExpired(['nvidia-smi'],10,stderr=b'driver stalled')
            with patch('src.weave.calibration.subprocess.run',side_effect=error):
                result=cal.capture_gpu_snapshot()
            path=Path(tmp)/'gpu-after.json'
            cal.atomic_json(path,result)
            value=json.loads(path.read_text())
            self.assertEqual(value['status'],'unavailable')
            self.assertEqual(value['stderr'],'driver stalled')

    def test_prelaunch_failure_does_not_inherit_previous_iteration_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=[120.]
            journal=cal.CalibrationJournal(Path(tmp),IDENTITY,'new-case',100.,1200.,clock=lambda:now[0])
            journal.begin(CASE,0,elapsed_seconds=20.,completed_cases=0)
            def launch(seconds): now[0]+=2.
            cal.execute_gpu_iteration(launch,lambda:None,journal=journal)
            now[0]=125.
            journal.begin(CASE,1,elapsed_seconds=25.,completed_cases=0)
            journal.fail(RuntimeError('prelaunch I/O failure'))
            failure=json.loads(journal.failure_path.read_text())
            self.assertIsNone(failure['gpu_iteration_elapsed_seconds'])
            self.assertIsNone(failure['gpu_iteration_started_monotonic'])

    def test_worker_accepts_host_snapshot_in_fresh_diagnostic_directory(self):
        from scripts.calibrate_weave import parse_args
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'diagnostic';out.mkdir()
            for name in ('launch.json','calibration.log','gpu-before.json'):
                (out/name).write_text('{}')
            try:
                args=parse_args(['--model-path',str(root/'model'),'--output-dir',str(out),
                    '--core-run-dir',str(root/'core'),'--diagnose-case','2048,131,1,compute'])
            except SystemExit: self.fail('host-written snapshot must not reject a fresh diagnostic launch')
            self.assertEqual(args.diagnose_case,CASE)
            (out/'timings-rank0.json').write_text('{}')
            with patch('sys.stderr'),self.assertRaises(SystemExit):
                parse_args(['--model-path',str(root/'model'),'--output-dir',str(out),
                    '--core-run-dir',str(root/'core'),'--diagnose-case','2048,131,1,compute'])

    def test_gpu_snapshot_uses_lease_devices_and_excludes_process_details(self):
        capture=getattr(cal,'capture_gpu_snapshot',None)
        self.assertTrue(callable(capture),'retain clock and ECC context for device faults')
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'nvidia-smi'
            xml=('<nvidia_smi_log><gpu id="0000:81:00.0"><uuid>GPU-lease4</uuid>'
                 '<product_name>H100</product_name><clocks><sm_clock>1590 MHz</sm_clock></clocks>'
                 '<temperature><gpu_temp>50 C</gpu_temp></temperature>'
                 '<processes><process_name>private-process</process_name></processes>'
                 '</gpu></nvidia_smi_log>')
            binary.write_text('#!'+sys.executable+'\nimport sys\n'
                'if sys.argv[1:]!=["-q","-x","-i","4,5,6,7"]: sys.exit(7)\n'
                'print('+repr(xml)+')\n')
            binary.chmod(0o755)
            with patch.dict(os.environ,{'PATH':tmp+os.pathsep+os.environ['PATH']}):
                result=capture()
            self.assertEqual(result['status'],'captured')
            self.assertEqual(result['devices'][0]['sm_clock'],'1590 MHz')
            self.assertEqual(result['devices'][0]['uuid'],'GPU-lease4')
            self.assertNotIn('private-process',json.dumps(result))

    def test_failure_retains_iteration_time_and_distinguishes_resumed_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=[120.]
            try:
                journal=cal.CalibrationJournal(Path(tmp),IDENTITY,'resume-362',100.,1200.,
                    clock=lambda:now[0],initial_completed_cases=362)
            except TypeError: self.fail('journal must retain invocation starting point count')
            journal.begin(CASE,1,elapsed_seconds=20.,completed_cases=362)
            def launch(seconds):
                self.assertEqual(seconds,120.)
                now[0]+=11.
            def sync(): raise RuntimeError('CUDA launch failure')
            with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}):
                with self.assertRaises(RuntimeError):
                    cal.execute_gpu_iteration(launch,sync,journal=journal,kernel_timeout_seconds=120.)
            failure=json.loads(journal.failure_path.read_text())
            self.assertEqual(failure['initial_completed_cases'],362)
            self.assertEqual(failure['committed_this_invocation'],0)
            self.assertEqual(failure['gpu_iteration_elapsed_seconds'],11.)
            self.assertEqual(failure['kernel_timeout_seconds'],120.)
            self.assertFalse(failure['launch_blocking'])

    def test_export_preflight_uses_work_disk_and_keeps_destination_fresh(self):
        prepare=getattr(exp,'prepare_export_directory',None)
        self.assertTrue(callable(prepare),'custom work-root must not export into the full NAS checkout')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp).resolve()/'work'
            destination=prepare(work,'full-experiment-test')
            self.assertEqual(destination,work/'public-reports/full-experiment-test')
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.iterdir()),[])
            destination.mkdir()
            with self.assertRaises(FileExistsError): prepare(work,'full-experiment-test')

    def test_export_quota_failure_is_reported_before_creating_run_destination(self):
        prepare=getattr(exp,'prepare_export_directory',None)
        self.assertTrue(callable(prepare))
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp).resolve()/'work'
            with patch('src.weave.experiment.tempfile.NamedTemporaryFile',side_effect=OSError(errno.EDQUOT,'quota')):
                with self.assertRaisesRegex(RuntimeError,'export-root'):
                    prepare(work,'full-experiment-failed')
            self.assertFalse((work/'public-reports/full-experiment-failed').exists())

    def test_full_controller_forwards_guard_without_changing_deadline(self):
        parse=getattr(full,'parse_args',None)
        build=getattr(full,'calibration_command',None)
        self.assertTrue(callable(parse) and callable(build),'full recovery needs an explicit calibration wait guard')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp).resolve()
            args,_=parse(['--work-root',str(work),'--calibration-kernel-timeout-seconds','120'])
            command=build(args,work/'core',work/'cal',work/'lifecycle.json','new')
            self.assertEqual(command[command.index('--kernel-timeout-seconds')+1],'120.0')
            path=work/'progress.json'
            path.write_text(json.dumps({'invocation_id':'new','measuring_started_monotonic':100.,'budget_seconds':1200}))
            self.assertEqual(full.measuring_deadline(path,'new'),1300.)
            for value in ('0','nan','1201'):
                with patch('sys.stderr'),self.assertRaises(SystemExit):
                    parse(['--calibration-kernel-timeout-seconds',value])

    def test_async_diagnostic_reaches_docker_and_records_selected_guard(self):
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
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_calibration_job.py'),
                '--model-path',str(root/'model'),'--work-root',str(root),
                '--core-run-dir',str(root/'core'),'--output-dir',str(root/'diagnostic'),
                '--lifecycle-path',str(root/'lifecycle.json'),'--export-dir',str(root/'public'),
                '--diagnose-case','2048,131,1,compute','--launch-blocking','0',
                '--kernel-timeout-seconds','120'],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,1,run.stderr)
            self.assertTrue(capture.is_file(),run.stderr)
            args=json.loads(capture.read_text())
            self.assertIn('CUDA_LAUNCH_BLOCKING=0',args)
            self.assertIn('--pull=never',args)
            self.assertEqual(args[args.index('--kernel-timeout-seconds')+1],'120.0')
            report=json.loads((root/'public/experiment.json').read_text())
            self.assertFalse(report['launch_blocking'])
            self.assertEqual(report['kernel_timeout_seconds'],120.)
            self.assertEqual(report['performance'],'not_run')
