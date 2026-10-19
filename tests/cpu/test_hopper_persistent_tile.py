"""Portable coverage test for the coordinates used by the device tile stores."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HopperPersistentTileTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
    def test_output_fragment_ownership_and_shared_alignment(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "hopper_persistent_tile"
            result = subprocess.run([
                "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I",
                str(root / "experimental/hopper_persistent"),
                str(root / "tests/cpp/test_hopper_persistent_tile.cpp"),
                "-o", str(exe),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
