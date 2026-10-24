"""CPU specifications for the real CUDA planner's candidates and finalize slices."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class ParallelTailTests(unittest.TestCase):
    def test_candidate_search_and_finalize_partition_match_independent_oracles(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'tail-parallel'
            result=subprocess.run([compiler,'-std=c++17','-O2','-pthread','-ffp-contract=off','-I',str(ROOT),
                str(ROOT/'tests/cpu/native_tail_parallel_host.cpp'),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(binary)],check=True,timeout=30)

    def test_finalize_coverage_rejects_missing_duplicate_or_early_publication(self):
        from src.weave.native_tail_executor import finalize_coverage_valid
        state=dict(finalize_tasks_per_chunk=64,finalize_claimed=[64,64,0,0],
                   finalize_completed=[64,64,0,0],finalized_chunks=2)
        self.assertTrue(finalize_coverage_valid(state,32,4096,2,128))
        for extra in (dict(finalize_completed=[64,63,0,0]),dict(finalize_claimed=[65,64,0,0]),
                      dict(finalized_chunks=1),dict(finalize_tasks_per_chunk=32),
                      dict(finalize_completed=[64,64,1,0])):
            self.assertFalse(finalize_coverage_valid(dict(state,**extra),32,4096,2,128))
