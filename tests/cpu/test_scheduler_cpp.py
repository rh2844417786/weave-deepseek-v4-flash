from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SharedSchedulerTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_device_scheduler_header_on_host(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'scheduler_test'
            subprocess.run(['c++', '-std=c++17', '-I', str(root/'csrc'), str(root/'tests/cpp/test_scheduler.cpp'), '-o', str(exe)], check=True, capture_output=True)
            subprocess.run([str(exe)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
