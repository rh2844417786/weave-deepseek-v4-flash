"""CPU checks for the real-Marlin GPU probe's routing and evidence contract."""
from contextlib import redirect_stderr
import io
import unittest

from tests.gpu import check_native_budget as probe


class NativeBudgetGPUContract(unittest.TestCase):
    def test_mapped_routing_preserves_partial_topk_and_empty_local_case(self):
        rows = probe.route_rows('balanced', 32)
        self.assertTrue(all(len(row) == 6 for row in rows))
        self.assertTrue(all(0 < sum(v >= 0 for v in row) < 6 for row in rows))
        self.assertTrue(all(len(set(v for v in row if v >= 0)) ==
                            sum(v >= 0 for v in row) for row in rows))
        self.assertEqual(probe.group_counts(probe.route_rows('empty', 32), 8),
                         {'global_blocks': 24, 'local_blocks': 0, 'local_rows': 0,
                          'total_rows': 192})

    def test_group_counts_include_remote_minus_one_padding(self):
        self.assertEqual(probe.group_counts([[0, -1, 1], [0, -1, -1]], 8),
                         {'global_blocks': 3, 'local_blocks': 2, 'local_rows': 3,
                          'total_rows': 6})

    def test_same_capture_schedule_returns_to_previous_grouping(self):
        schedule = probe.replay_schedule(3)
        self.assertGreaterEqual(schedule.count('empty'), 3)
        self.assertGreaterEqual(schedule.count('heavy'), 6)
        for batch in (7, 32):
            workloads = [probe.group_counts(probe.route_rows(name, batch), 8)
                         for name in schedule]
            self.assertGreater(len({item['local_rows'] for item in workloads}), 2)
            self.assertEqual(workloads[0], workloads[-1])

    def test_cli_requires_real_checkpoint_and_positive_replay_cycles(self):
        args = probe.parse_args(['--model-path', '/model', '--output-dir', '/out',
                                 '--cache-root', '/cache'])
        self.assertEqual(args.layer, 0)
        self.assertEqual(args.replay_cycles, 3)
        self.assertEqual(args.batches, [7, 32])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            probe.parse_args(['--model-path', '/model', '--output-dir', '/out',
                              '--cache-root', '/cache', '--replay-cycles', '0'])

    def test_state_contract_checks_actual_replay_and_empty_bucket(self):
        before = [0] * 19
        after = [1, 1, 132, 0, 24, 0, 32, 16] + [1] + [0] * 8 + [0, 192]
        expected = probe.group_counts(probe.route_rows('empty', 32), 8)
        state = probe.validate_state(before, after, expected, 1)
        self.assertEqual(state['active_workers'], 0)
        with self.assertRaisesRegex(RuntimeError, 'call count'):
            probe.validate_state(after, after, expected, 1)
        bad_histogram = list(after)
        bad_histogram[8], bad_histogram[9] = 0, 1
        with self.assertRaisesRegex(RuntimeError, 'histogram'):
            probe.validate_state(before, bad_histogram, expected, 1)

    def test_full_mode_uses_native_grid_but_evidence_is_not_native_baseline(self):
        work = probe.group_counts(probe.route_rows('heavy', 32), 8)
        after = [1, 0, 132, 132, work['global_blocks'], work['local_blocks'], 32, 16]
        after += [0] * 8 + [1] + [192, 192]
        self.assertEqual(probe.validate_state([0] * 19, after, work, 2)['native_grid'], 132)
        after[3] = 99
        with self.assertRaisesRegex(RuntimeError, 'native grid'):
            probe.validate_state([0] * 19, after, work, 2)


if __name__ == '__main__':
    unittest.main()
