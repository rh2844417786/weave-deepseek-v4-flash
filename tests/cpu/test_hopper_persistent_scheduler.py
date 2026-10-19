"""Exercise actual portable M64 cost and concurrent ready-claim functions."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HopperPersistentSchedulerTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
    def test_cost_and_concurrent_dependency_claims(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "hopper_persistent_scheduler"
            result = subprocess.run([
                "c++", "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                str(root / "tests/cpp/test_hopper_persistent_scheduler.cpp"),
                "-o", str(exe),
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
