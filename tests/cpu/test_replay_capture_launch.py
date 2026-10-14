import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ReplayCaptureLaunchTest(unittest.TestCase):
    def test_driver_preview_uses_installed_benchmark_without_profile_reset_and_accepts_long_deadlines(self):
        result=subprocess.run([sys.executable,'scripts/run_replay_capture.py','--work-root','/tmp/weave-work',
            '--dataset','/tmp/existing9x200.jsonl','--core-run-dir','/tmp/weave-work/current-core',
            '--setup-timeout','18000','--capture-timeout','18000'],cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7'},text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout)
        cmd=value['load_command']
        self.assertIn('--installed-benchmark',cmd)
        self.assertEqual(cmd[cmd.index('--max-concurrency')+1],'32')
        self.assertEqual(cmd[cmd.index('--sharegpt-output-len')+1],'1024')
        self.assertNotIn('/workspace/scripts/bench_c32_profile.py',cmd)
        self.assertFalse(value['used_for_formal_performance'])

    def run_launch(self,*extra):
        return subprocess.run([sys.executable,'scripts/run_native_ep.py','--work-root','/tmp/weave-work',
            '--backend','weave_functional','--kernel','tiled','--replay-capture-dir','/tmp/weave-work/private-trace',
            '--replay-dataset-sha256','b'*64,'--checkpoint-content-sha256','a'*64,
            '--replay-tokenizer-manifest-sha256','c'*64,
            '--replay-capture-id','11111111-1111-4111-8111-111111111111',*extra],
            cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7','WEAVE_MAX_RUNNING_REQUESTS':'32'},
            text=True,capture_output=True)

    def test_capture_launch_is_private_offline_fixed_eager_with_cached_control(self):
        result=self.run_launch('--no-hot-cache')
        self.assertEqual(result.returncode,0,result.stderr)
        cmd=json.loads(result.stdout)['docker_argv']
        self.assertIn('WEAVE_REPLAY_CAPTURE_DIR=/work/private-trace',cmd)
        self.assertIn('WEAVE_REPLAY_DATASET_SHA256='+'b'*64,cmd)
        self.assertIn('WEAVE_REPLAY_TOKENIZER_MANIFEST_SHA256='+'c'*64,cmd)
        self.assertIn('WEAVE_REPLAY_CAPTURE_ID=11111111-1111-4111-8111-111111111111',cmd)
        self.assertIn('WEAVE_HOT_CACHE=0',cmd)
        self.assertIn('HF_HUB_OFFLINE=1',cmd)
        self.assertIn('--pull=never',cmd)
        self.assertEqual(cmd[cmd.index('--cuda-graph-backend-decode')+1],'disabled')

    def test_capture_requires_tokenizer_and_canonical_uuid_identity(self):
        for extra in [('--replay-tokenizer-manifest-sha256','invalid'),('--replay-capture-id','not-a-uuid'),
                      ('--replay-capture-id','11111111-1111-1111-8111-111111111111')]:
            result=self.run_launch(*extra)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('capture',result.stderr)

    def test_capture_rejects_graph_profile_and_directory_outside_work(self):
        for extra in [('--decode-graph','full'),('--profile','/tmp/weave-work/bank'),
                      ('--execution-profile',),('--replay-capture-dir','/tmp/other-trace')]:
            result=self.run_launch(*extra)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('capture',result.stderr)


if __name__=='__main__':unittest.main()
