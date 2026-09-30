"""Model the Docker daemon state surviving death of the foreground CLI."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE_DOCKER = '''#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
state=Path(os.environ['CONTAINER_STATE'])
args=sys.argv[1:]
if args[:2]==['image','inspect']:
 print('sha256:fixture-image')
elif args[0]=='run':
 name=args[args.index('--name')+1]
 state.write_text(json.dumps({'name':name,'running':True}))
 while state.exists() and json.loads(state.read_text())['running']: time.sleep(.05)
elif args[0] in ('stop','rm'):
 data=json.loads(state.read_text()) if state.exists() else {'name':None,'running':False}
 if data['name'] != args[-1]: sys.exit(3)
 if args[0]=='stop' and os.environ.get('STOP_HANG'): time.sleep(20)
 data['running']=False;state.write_text(json.dumps(data))
elif args[0]=='inspect':
 print('true' if state.exists() and json.loads(state.read_text())['running'] else 'false')
'''


class ContainerLifecycleTest(unittest.TestCase):
    def exercise(self, signum, hang=False, native=False):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            docker=work/'docker'
            docker.write_text(FAKE_DOCKER);docker.chmod(0o755)
            state=work/'state.json'
            env={**os.environ,'PATH':str(work)+os.pathsep+os.environ['PATH'],
                 'CONTAINER_STATE':str(state),'CUDA_VISIBLE_DEVICES':'4,5,6,7'}
            if hang: env['STOP_HANG']='1'
            else: env.pop('STOP_HANG',None)
            if native:
                model=work/'model';model.mkdir()
                command=[sys.executable,'scripts/run_native_ep.py','--work-root',str(work),
                         '--model-path',str(model),'--execute']
            else:
                command=[sys.executable,'scripts/manage_container.py','--name','weave-test-owned',
                         '--evidence',str(work/'lifecycle.json'),'--','docker','run','--rm',
                         '--name','weave-test-owned','fixture-image']
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:
                deadline=time.monotonic()+5
                while not state.exists():
                    if process.poll() is not None:
                        self.fail(process.communicate()[1])
                    if time.monotonic()>deadline: self.fail('container did not start')
                    time.sleep(.02)
                started=time.monotonic()
                os.killpg(process.pid,signum)
                stdout,stderr=process.communicate(timeout=9)
                self.assertEqual(process.returncode,128+signum,stderr)
                self.assertFalse(json.loads(state.read_text())['running'])
                self.assertLess(time.monotonic()-started,9)
                evidence=next(work.glob('native-ep-*/container_lifecycle.json')) if native else work/'lifecycle.json'
                data=json.loads(evidence.read_text())
                self.assertEqual(data['termination_signal'],signum)
                self.assertEqual(data['cleanup_status'],'stopped')
                if native:
                    report=json.loads(next(work.glob('native-ep-*/launch.json')).read_text())
                    self.assertEqual(report['exit_code'],143)
                    self.assertEqual(report['termination_signal'],signal.SIGTERM)
            finally:
                if state.exists():
                    data=json.loads(state.read_text());data['running']=False;state.write_text(json.dumps(data))
                if process.poll() is None: os.killpg(process.pid,signal.SIGKILL)
                process.communicate(timeout=3)

    def test_gpuq_sigterm_cleans_native_container_and_records_exit(self):
        self.exercise(signal.SIGTERM,native=True)

    def test_sigint_cleans_managed_container(self):
        self.exercise(signal.SIGINT)

    def test_stop_timeout_falls_back_to_force_removal_before_gpuq_kill(self):
        self.exercise(signal.SIGTERM,hang=True)

    def test_mismatched_container_name_is_rejected(self):
        run=subprocess.run([sys.executable,'scripts/manage_container.py','--name','weave-owned',
                            '--evidence','/tmp/unused.json','--','docker','run','--rm',
                            '--name','some-other-job','fixture'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('name',run.stderr)


if __name__=='__main__': unittest.main()
