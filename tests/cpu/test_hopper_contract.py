"""Run the same portable helpers used by CUDA, without importing Torch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HopperContractTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
    def test_nibble_swizzle_quant_scale_and_online_route_contract(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "hopper_contract"
            result = subprocess.run([
                "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I",
                str(root / "experimental/hopper_fp8"),
                str(root / "tests/cpp/test_hopper_contract.cpp"), "-o", str(exe),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
