from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ExpertNumericsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_cuda_shared_arithmetic_obeys_native_bf16_boundaries(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder)/'expert_numerics'
            build = subprocess.run(['c++', '-std=c++17', '-I', str(root/'csrc'),
                str(root/'tests/cpp/test_expert_numerics.cpp'), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
