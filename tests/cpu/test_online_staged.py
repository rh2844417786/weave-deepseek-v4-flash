from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from src.weave.staged_plan import build_staged_groups


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
class OnlineStagedPlannerTest(unittest.TestCase):
    def test_gpu_shared_planner_preserves_p1_order_and_safe_capacity(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'online_staged'
            result = subprocess.run(['c++', '-std=c++17', '-I', str(root / 'csrc'),
                                     str(root / 'tests/cpp/test_online_staged.cpp'), '-o', str(exe)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rng = random.Random(241014)
            for batch in (1, 7, 16, 17, 31, 32):
                for experts, topk in ((8, 6), (256, 6), (64, 16)):
                    for dense in (False, True):
                        ids = ([list(range(topk)) for _ in range(batch)] if dense else
                               [rng.sample(range(experts), topk) for _ in range(batch)])
                        for rank in range(4):
                            payload = ' '.join(map(str, [batch, topk, rank, 4, experts] +
                                                   [x for row in ids for x in row]))
                            run = subprocess.run([str(exe), 'plan'], input=payload,
                                                 capture_output=True, text=True)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            values = list(map(int, run.stdout.split()))
                            capacity, count = values[:2]
                            self.assertGreaterEqual(count, 0)
                            self.assertLessEqual(capacity, batch * topk)
                            oracle = build_staged_groups(ids, rank=rank, ep=4, experts=experts,
                                                         topk=topk, capacity=capacity)
                            self.assertEqual(count, oracle.group_count)
                            self.assertEqual(values[2:2+2*capacity],
                                             [x for row in oracle.group_meta for x in row])
                            self.assertEqual(values[2+2*capacity:],
                                             [x for row in oracle.group_slots for x in row])


if __name__ == '__main__':
    unittest.main()
