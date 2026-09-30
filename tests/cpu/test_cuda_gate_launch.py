import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def copy_manager(repo):
    (repo/'src/weave').mkdir(parents=True)
    shutil.copy(ROOT/'scripts/manage_container.py',repo/'scripts')
    shutil.copy(ROOT/'src/weave/container_lifecycle.py',repo/'src/weave')


class CUDAGateLauncherTest(unittest.TestCase):
    def test_default_work_directory_and_resolved_image_survive_gpuq_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)/'repo'
            (repo/'scripts').mkdir(parents=True)
            (repo/'configs').mkdir()
            shutil.copy(ROOT/'scripts/run_cuda_functional_gpuq.sh', repo/'scripts')
            shutil.copy(ROOT/'configs/runtime.json', repo/'configs')
            copy_manager(repo)
            bin_dir = Path(tmp)/'bin'
            bin_dir.mkdir()
            capture = Path(tmp)/'docker_args.json'
            docker = bin_dir/'docker'
            docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'if sys.argv[1:3] == ["image","inspect"]: print("sha256:actual-image")\n'
                'elif sys.argv[1] == "run":\n'
                '    open(os.environ["DOCKER_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n'
                '    sys.exit(7)\n'
                'else:\n'
                '    print("No such object",file=sys.stderr);sys.exit(1)\n')
            docker.chmod(0o755)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'DOCKER_CAPTURE': str(capture),
                   'PATH': str(bin_dir)+os.pathsep+os.environ['PATH']}
            env.pop('WORK_ROOT', None)
            env.pop('IMAGE', None)
            run = subprocess.run(['bash', str(repo/'scripts/run_cuda_functional_gpuq.sh')],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 7, run.stderr)
            args = json.loads(capture.read_text())
            self.assertEqual(args[args.index('--gpus')+1], '"device=4,5,6,7"')
            self.assertIn('sha256:actual-image', args)
            self.assertIn(str(repo/'work')+':/work', args)
            self.assertIn('scripts/check_runtime.py', args)
            self.assertIn('--run-core-tests', args)
            run_dirs = list((repo/'work').glob('cuda-functional.*'))
            self.assertEqual(len(run_dirs), 1)
            self.assertEqual((run_dirs[0]/'exit_code.txt').read_text().strip(), '7')

    def test_preflight_uses_the_same_local_image_and_work_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)/'repo'
            (repo/'scripts').mkdir(parents=True)
            (repo/'configs').mkdir()
            shutil.copy(ROOT/'scripts/run_preflight_gpuq.sh', repo/'scripts')
            shutil.copy(ROOT/'configs/runtime.json', repo/'configs')
            copy_manager(repo)
            bin_dir = Path(tmp)/'bin'
            bin_dir.mkdir()
            capture = Path(tmp)/'args.json'
            docker = bin_dir/'docker'
            docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'if sys.argv[1:3] == ["image","inspect"]: print("sha256:actual-image")\n'
                'else: open(os.environ["DOCKER_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
            docker.chmod(0o755)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'DOCKER_CAPTURE': str(capture),
                   'PATH': str(bin_dir)+os.pathsep+os.environ['PATH']}
            for key in ('WORK_ROOT', 'IMAGE', 'OUTPUT_REL'):
                env.pop(key, None)
            run = subprocess.run(['bash', str(repo/'scripts/run_preflight_gpuq.sh')], env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            args = json.loads(capture.read_text())
            self.assertIn(str(repo/'work')+':/work', args)
            self.assertIn('lmsysorg/sglang:v0.5.17', args)
            self.assertEqual(args[args.index('--runtime-image-id')+1], 'sha256:actual-image')
            self.assertTrue(args[args.index('--output')+1].startswith('/work/preflight.'))


if __name__ == '__main__':
    unittest.main()
