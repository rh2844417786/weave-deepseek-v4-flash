from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TileSchedulerTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_tile_claims_and_dependencies_on_host(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'tile_scheduler'
            build=subprocess.run(['c++','-std=c++17','-I',str(root/'csrc'),
                str(root/'tests/cpp/test_tile_scheduler.cpp'),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
