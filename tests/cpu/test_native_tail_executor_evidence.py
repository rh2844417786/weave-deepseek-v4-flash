import unittest
from tests.gpu import check_native_tail_executor as probe


class ExecutorEvidenceTests(unittest.TestCase):
    def test_envelopes_do_not_count_as_actual_overlap(self):
        self.assertFalse(probe.any_interval_overlap([(10, 20), (100, 110)], [(40, 50)]))
        self.assertTrue(probe.any_interval_overlap([(10, 20), (100, 110)], [(18, 22)]))
        self.assertFalse(probe.any_interval_overlap([(10, 20)], [(20, 30)]))

    def test_exact_bitmap_rejects_same_total_wrong_tasks(self):
        with self.assertRaisesRegex(RuntimeError, 'exact down task'):
            probe.check_visits([2, 0, 0], [0, 1])
        with self.assertRaisesRegex(RuntimeError, 'exact down task'):
            probe.check_visits([1, 0, 1], [0, 1])
        probe.check_visits([1, 1, 0], [0, 1])

    def test_role_recovery_requires_same_worker_and_order(self):
        roles = [[1], [0], [1]]
        self.assertTrue(probe.stolen_then_communication(roles, [(0, 10, 20)], [(0, 30, 40)]))
        self.assertFalse(probe.stolen_then_communication(roles, [(0, 10, 20)], [(2, 30, 40)]))
        self.assertFalse(probe.stolen_then_communication(roles, [(1, 10, 20)], [(1, 30, 40)]))
        self.assertFalse(probe.stolen_then_communication(roles, [(0, 30, 40)], [(0, 10, 20)]))

    def test_full_task_trace_checks_roles_counts_and_real_intervals(self):
        events = [[1, 10, 20, 1], [1, 100, 110, 1]]
        events += [[0, 40 + task, 41 + task, 2 if task < 2 else 3] for task in range(10)]
        events += [[0, 0, 0, 0] for _ in range(10)]
        snapshot = dict(visits=[1, 1], events=events,
            trace=[[1, 3, 1, 200, 0, 10, 0, 39, 40, 0], [0, 4, 1, 200, 2, 0, 0, 0, 0, 0]])
        kwargs = dict(tiles=2, capacity=1, plan=dict(chunks=2, communication_workers=1))
        result = probe.analyze_trace(snapshot, dict(expected_task_ids=[0, 1]), **kwargs)
        self.assertFalse(result['overlap_observed'])
        self.assertEqual(result['local_down_tasks'], 2)
        snapshot['events'][2][1:3] = [101, 105]
        self.assertTrue(probe.analyze_trace(snapshot, dict(expected_task_ids=[0, 1]), **kwargs)['overlap_observed'])
        snapshot['trace'][0][5] = 9
        with self.assertRaisesRegex(RuntimeError, 'worker counters'):
            probe.analyze_trace(snapshot, dict(expected_task_ids=[0, 1]), **kwargs)


if __name__ == '__main__':
    unittest.main()
