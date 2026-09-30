import unittest
from src.weave.cost_model import Workload, Candidate, select_plan


class CostModelTest(unittest.TestCase):
    def test_joint_search_balances_chunk_efficiency_and_tail(self):
        work = Workload(6, 4, 2, 10, 2, 2)
        candidates = [Candidate(1, 1, 120, 20, 1, .5),
                      Candidate(1, 2, 120, 20, .9, .5),
                      Candidate(1, 4, 120, 20, .5, .5)]
        p = select_plan(work, 4, candidates)
        self.assertEqual((p.comm_workers, p.chunks, p.steal_tasks), (1, 2, 0))
        self.assertAlmostEqual(p.seconds, 240 / 108 + .4)

    def test_steal_budget_uses_residual_work(self):
        p = select_plan(Workload(8, 2, 1, 10, 2, 2), 4, [Candidate(1, 1, 24, 120, 1, 0)])
        self.assertEqual(p.steal_tasks, 2)

    def test_zero_comm_only_legal_without_remote_pairs(self):
        with self.assertRaises(ValueError):
            select_plan(Workload(0, 1, 1, 1, 2, 2), 4, [Candidate(0, 1, 24, 0, 1, 0)])
        p = select_plan(Workload(1, 0, 0, 1, 2, 2), 4, [Candidate(0, 1, 24, 0, 1, 0)])
        self.assertEqual(p.seconds, 1)

    def test_no_fabricated_profile_and_invalid_rates_rejected(self):
        for rows in ([], [Candidate(1, 1, 0, 20, 1, 0)], [Candidate(1, 1, 24, 20, 0, 0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_plan(Workload(1, 1, 1, 2, 2, 2), 4, rows)


if __name__ == '__main__':
    unittest.main()
