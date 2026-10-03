"""The launch boundary must preserve the four allocated GPUs and EP settings."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeEPLaunchTest(unittest.TestCase):
    def test_tiled_profile_and_timing_are_explicit_in_formal_launch(self):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7'}
        for backend in ('native','weave_functional'):
            command=[sys.executable,'scripts/run_native_ep.py','--backend',backend,
                     '--work-root','/tmp/weave-work','--request-timing','--decode-graph','full']
            if backend=='weave_functional':
                command += ['--kernel','tiled','--profile','/tmp/weave-work/profiles',
                            '--checkpoint-content-sha256','a'*64]
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=json.loads(result.stdout)['docker_argv']
            self.assertIn('WEAVE_REQUEST_TIMING=1',args)
            self.assertIn('PYTHONPATH=/workspace/timing_hooks:/workspace/src',args)
            self.assertIn('--incremental-streaming-output',args)
            self.assertEqual(args[args.index('--stream-interval')+1],'1')
            if backend=='weave_functional':
                self.assertIn('WEAVE_KERNEL=tiled',args)
                self.assertIn('WEAVE_PROFILE=/work/profiles',args)
                self.assertIn('WEAVE_CHECKPOINT_CONTENT_SHA256='+'a'*64,args)

    def test_wmma_graph_candidate_keeps_ep4_and_uses_explicit_graph_opt_in(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        run = subprocess.run([sys.executable, 'scripts/run_native_ep.py', '--backend', 'weave_functional',
                              '--kernel', 'wmma', '--decode-graph', 'full'], cwd=ROOT,
                             env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        args = json.loads(run.stdout)['docker_argv']
        self.assertIn('WEAVE_KERNEL=wmma', args)
        self.assertIn('WEAVE_DECODE_GRAPH=1', args)
        self.assertIn('WEAVE_EVIDENCE_PORT_BASE=30110', args)
        self.assertEqual(args[args.index('--cuda-graph-backend-decode')+1], 'full')
        self.assertEqual(args[args.index('--cuda-graph-max-bs-decode')+1], '1')
        self.assertEqual(args[args.index('--ep-size')+1], '4')

    def test_default_image_and_work_root_follow_current_host_contract(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        env.pop('WORK_ROOT', None)
        run = subprocess.run([sys.executable, 'scripts/run_native_ep.py'], cwd=ROOT,
                             env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        args = json.loads(run.stdout)['docker_argv']
        self.assertIn('lmsysorg/sglang:v0.5.17', args)
        self.assertIn(str(ROOT/'work') + ':/work', args)
        self.assertIn('/workspace/scripts/check_runtime.py', args)

    def test_functional_backend_mounts_extension_and_selects_eager_explicitly(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        run = subprocess.run([sys.executable, 'scripts/run_native_ep.py', '--work-root', '/tmp/weave-work',
                              '--backend', 'weave_functional'], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        args = json.loads(run.stdout)['docker_argv']
        self.assertIn('SGLANG_EXTERNAL_MODEL_PACKAGE=weave.sglang_ext.models', args)
        self.assertIn('PYTHONPATH=/workspace/src', args)
        self.assertEqual(args[args.index('--cuda-graph-backend-decode')+1], 'disabled')
        self.assertEqual(args[args.index('--ep-size')+1], '4')

    def test_preview_is_ep4_and_docker_device_list_is_one_quoted_argument(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        run = subprocess.run([sys.executable, 'scripts/run_native_ep.py', '--work-root', '/tmp/weave-work'],
                             cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        data = json.loads(run.stdout)
        args = data['docker_argv']
        self.assertEqual(args[args.index('--gpus') + 1], '"device=4,5,6,7"')
        self.assertEqual(args[args.index('--ep-size') + 1], '4')
        self.assertEqual(args[args.index('--tp-size') + 1], '4')
        self.assertEqual(args[args.index('--max-running-requests') + 1], '1')
        self.assertIn('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash:/model:ro', args)
        self.assertEqual(data['status'], 'preview_not_executed')

    def test_rejects_gpu_allocation_outside_authorized_cards(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3'}
        run = subprocess.run([sys.executable, 'scripts/run_native_ep.py', '--work-root', '/tmp/weave-work'],
                             cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('4,5,6,7', run.stderr)

    def test_execution_records_resolved_image_and_exit_without_claiming_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            docker = work / 'docker'
            docker.write_text('#!/usr/bin/env python3\nimport sys\n'
                              'if sys.argv[1:3] == ["image", "inspect"]: print("sha256:fixture-image")\n'
                              'elif sys.argv[1] == "run":\n'
                              '    assert "sha256:fixture-image" in sys.argv\n'
                              '    sys.exit(7)\n')
            docker.chmod(0o755)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'PATH': str(work) + os.pathsep + os.environ['PATH']}
            model = work / 'model'
            model.mkdir()
            run = subprocess.run([sys.executable, 'scripts/run_native_ep.py', '--work-root', str(work),
                                  '--model-path', str(model), '--execute'], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 7, run.stderr)
            reports = list(work.glob('native-ep-*/launch.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['resolved_image_id'], 'sha256:fixture-image')
            self.assertEqual(report['exit_code'], 7)
            self.assertEqual(report['generation_status'], 'not_verified')


if __name__ == '__main__':
    unittest.main()
