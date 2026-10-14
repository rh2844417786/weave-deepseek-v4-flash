from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from src.weave.staged_plan import build_staged_groups


class StagedPlanTest(unittest.TestCase):
    def test_local_ownership_preserves_original_slot_positions_and_padding(self):
        plan = build_staged_groups([[0, 7, 2], [7, 1, 2], [4, 2, 6]],
                                   rank=0, ep=2, experts=8, topk=3, capacity=5)
        self.assertEqual(plan.group_count, 3)
        self.assertEqual(plan.capacity, 5)
        self.assertEqual(plan.local_pairs, 5)
        self.assertEqual(plan.group_meta, ((0, 1), (2, 3), (1, 1), (-1, 0), (-1, 0)))
        self.assertEqual(plan.group_slots[0], (0,) + (-1,) * 15)
        self.assertEqual(plan.group_slots[1], (2, 5, 7) + (-1,) * 13)
        self.assertEqual(plan.group_slots[2], (4,) + (-1,) * 15)
        self.assertEqual(plan.group_slots[3:], ((-1,) * 16,) * 2)

    def test_all_rank_groups_partition_every_original_route_once(self):
        ids = [[(token * 7 + slot * 5) % 32 for slot in range(6)] for token in range(32)]
        positions = []
        for rank in range(4):
            plan = build_staged_groups(ids, rank=rank, ep=4, experts=32)
            for (expert, rows), slots in zip(plan.group_meta, plan.group_slots):
                for pos in slots[:rows]:
                    token, slot = divmod(pos, 6)
                    self.assertEqual(ids[token][slot], rank * 8 + expert)
                    positions.append(pos)
        self.assertEqual(sorted(positions), list(range(192)))

    def test_dense_expert_splits_at_sixteen_without_dropping_tail(self):
        plan = build_staged_groups([[1, 7]] * 31, rank=0, ep=2, experts=8, topk=2)
        self.assertEqual(plan.group_meta, ((1, 16), (1, 15)))
        self.assertEqual(plan.group_slots[0], tuple(range(0, 32, 2)))
        self.assertEqual(plan.group_slots[1], tuple(range(32, 62, 2)) + (-1,))
        self.assertEqual(plan.local_pairs, 31)

    def test_empty_rank_has_a_safe_capacity_but_zero_active_groups(self):
        plan = build_staged_groups([[4, 5], [6, 7]], rank=0, ep=2, experts=8, topk=2)
        self.assertEqual(plan.group_count, 0)
        self.assertEqual(plan.group_meta, ((-1, 0),))
        self.assertEqual(plan.group_slots, ((-1,) * 16,))
        self.assertEqual(plan.local_pairs, 0)

    def test_full_group_rollover_follows_encounter_order(self):
        plan = build_staged_groups([[2, 0]] * 17, rank=0, ep=2, experts=8, topk=2)
        self.assertEqual(plan.group_meta, ((2, 16), (0, 16), (2, 1), (0, 1)))
        self.assertEqual(plan.group_slots[2][0], 32)
        self.assertEqual(plan.group_slots[3][0], 33)

    def test_invalid_and_duplicate_routes_fail_before_upload(self):
        for ids in ([[0, 0]], [[-1, 3]], [[8, 3]], [[0, 1.0]], [[True, 3]], [[0]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_staged_groups(ids, rank=0, ep=2, experts=8, topk=2)

    def test_invalid_shape_or_capacity_fails(self):
        for kwargs in ({'capacity': 0}, {'capacity': 1}, {'capacity': 5},
                       {'rank': 2}, {'ep': 3}, {'topk': 0}):
            options = dict(rank=0, ep=2, experts=8, topk=2)
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_staged_groups([[0, 1], [2, 3]], **options)
        for ids in ([], [[0, 1]] * 33):
            with self.subTest(batch=len(ids)), self.assertRaises(ValueError):
                build_staged_groups(ids, rank=0, ep=2, experts=8, topk=2)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_cuda_grid_helper_visits_every_tile_once(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'staged_grid'
            build = subprocess.run(['c++', '-std=c++17', '-I', str(root / 'csrc'),
                                    str(root / 'tests/cpp/test_staged_grid.cpp'), '-o', str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
