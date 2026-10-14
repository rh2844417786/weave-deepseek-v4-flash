import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]


class StagedLaunchTest(unittest.TestCase):
    def launch(self,*args,environment=None):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7','WEAVE_MAX_RUNNING_REQUESTS':'32'}
        env.pop('WEAVE_DECODE_IMPLEMENTATION',None)
        env.update(environment or {})
        return subprocess.run([sys.executable,'scripts/run_native_ep.py',*args],
                              cwd=ROOT,env=env,capture_output=True,text=True)

    def test_explicit_staged_eager_configuration_is_forwarded_into_container(self):
        result=self.launch('--backend','weave_functional','--kernel','tiled',
                           '--decode-implementation','staged_direct')
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['docker_argv']
        self.assertIn('WEAVE_DECODE_IMPLEMENTATION=staged_direct',command)
        self.assertIn('WEAVE_HOT_CACHE=1',command)
        self.assertEqual(command[command.index('--cuda-graph-backend-decode')+1],'disabled')

    def test_default_remains_persistent_and_environment_selects_staged(self):
        result=self.launch('--backend','weave_functional','--kernel','tiled')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('WEAVE_DECODE_IMPLEMENTATION=persistent',json.loads(result.stdout)['docker_argv'])
        result=self.launch('--backend','weave_functional','--kernel','tiled',
                           environment={'WEAVE_DECODE_IMPLEMENTATION':'staged_direct'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('WEAVE_DECODE_IMPLEMENTATION=staged_direct',json.loads(result.stdout)['docker_argv'])

    def test_invalid_staged_configurations_fail_at_preview(self):
        base=['--backend','weave_functional','--kernel','tiled','--decode-implementation','staged_direct']
        for extra in (['--decode-graph','full'],['--no-hot-cache'],['--fixed-plan','0,1,0'],
                      ['--profile','/tmp/bank'],['--execution-profile'],['--kernel','wmma'],
                      ['--backend','native'],['--replay-capture-dir','/tmp/capture']):
            with self.subTest(extra=extra):
                result=self.launch(*base,*extra)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('staged_direct',result.stderr)


if __name__=='__main__':unittest.main()
