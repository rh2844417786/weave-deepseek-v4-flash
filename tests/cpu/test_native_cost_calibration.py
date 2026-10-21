"""CPU coverage of calibration evidence and safe invocation preparation."""
from contextlib import redirect_stderr
import io
import unittest


class CalibrationContract(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('src.weave.native_cost_calibration'),
                             'actual native calibration implementation is required')
        from src.weave import native_cost_calibration
        return native_cost_calibration

    def test_timing_statistics_reject_empty_nonfinite_and_nonpositive_samples(self):
        module = self.module()
        self.assertEqual(module.timing_summary([1., 3., 2.]),
                         {'samples_us': [1., 3., 2.], 'median_us': 2.,
                          'mean_us': 2., 'stddev_us': (2 / 3) ** .5,
                          'min_us': 1., 'max_us': 3.})
        for values in ([], [0.], [-1.], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                module.timing_summary(values)

    def test_workload_changes_are_recorded_instead_of_rejected(self):
        from src.weave.native_cost import workload_difference
        reference = dict(features=[6,1,1,36,6], total_rows=6, global_blocks=1,
                         native_routing_sha256='a'*64)
        observed = dict(features=[3,2,2,5,2], total_rows=6, global_blocks=3,
                        native_routing_sha256='b'*64)
        result = workload_difference(reference, observed)
        self.assertEqual(result['policy'], 'record_only')
        self.assertFalse(result['features_match'])
        self.assertFalse(result['native_routing_sha256_match'])
        self.assertEqual(result['feature_delta'], [-3,1,1,-31,-4])
        self.assertTrue(workload_difference(reference, reference)['features_match'])

    def test_prediction_validation_records_unbounded_errors_and_regret(self):
        module = self.module()
        record = module.prediction_record([100., 1., 2.], [3., 10., 2.], 1)
        self.assertEqual(record['absolute_error_us'], [97., 9., 0.])
        self.assertEqual(record['relative_error'], [97 / 3, .9, 0.])
        self.assertEqual(record['selected_actual_us'], 10.)
        self.assertEqual(record['best_actual_us'], 2.)
        self.assertEqual(record['regret_us'], 8.)
        self.assertEqual(record['regret_fraction'], 4.)
        self.assertEqual(record['numerical_policy'], 'record_only')

    def test_gpu_choice_must_match_deterministic_spec_without_quality_cutoff(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'require_planner_selection'),
                        'calibration must check deterministic planner selection immediately')
        prediction = {'candidate_index': 1, 'predictions_us': [100., 1., 2.]}
        module.require_planner_selection(prediction, 1, 0)
        for selected, invalid_count in ((0, 0), (1, 1), (-1, 0)):
            with self.assertRaises(RuntimeError):
                module.require_planner_selection(prediction, selected, invalid_count)

    def test_zero_event_overhead_is_a_valid_clock_resolution_observation(self):
        from types import SimpleNamespace
        module = self.module()
        class Event:
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self, other): return 0.
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Event()))
        self.assertEqual(module._event_overhead(torch, 2),
                         {'samples_us': [0., 0.], 'median_us': 0.})

    def test_private_signature_binds_allocator_and_keeps_original_batch(self):
        from types import SimpleNamespace
        module = self.module()
        class Tensor:
            def __init__(self, label): self.label = label
            def clone(self): return Tensor(self.label + '_private')
        hidden, ids = Tensor('hidden'), Tensor('ids')
        batch = SimpleNamespace(input_ids=ids, other='context')
        def native(hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                   input_ids=None, input_ids_global=None, skip_shared_experts=False): pass
        bound = module.private_arguments(native, (hidden, batch, 'shared_allocator'),
                                         {'input_ids': ids, 'input_ids_global': ids})
        self.assertIsNot(bound.arguments['hidden_states'], hidden)
        self.assertIsNone(bound.arguments['gemm_output_zero_allocator'])
        self.assertIsNot(bound.arguments['forward_batch'], batch)
        self.assertIs(batch.input_ids, ids)
        self.assertIsNot(bound.arguments['input_ids'], ids)
        self.assertEqual(bound.arguments['forward_batch'].other, 'context')

    def test_gpu_cli_requires_measured_table_and_preserves_replay_case_controls(self):
        from pathlib import Path
        self.module()
        from tests.gpu import check_native_cost as probe
        base = ['--model-path', '/model', '--output-dir', '/out', '--cache-root', '/cache']
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            probe.parse_args(base)
        args = probe.parse_args(base + ['--table-dir', '/tables'])
        self.assertEqual(args.table_dir, Path('/tables'))
        self.assertEqual(args.batches, [7, 32])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            probe.parse_args(base + ['--table-dir', '/tables', '--batches', '32', '7'])

    def test_gpu_plan_probe_detects_feature_histogram_and_invalid_flag_corruption(self):
        from tests.gpu import check_native_cost as probe
        before = [0] * 32
        after = [8, 8, 3, 2, 3, 6, 2, 5, 2, 5000, 1] + [1] + [0] * 20
        rows = [[0, -1, 1], [0, -1, -1]]
        self.assertEqual(probe.validate_plan(before, after, rows, 8)['features'], [3, 2, 2, 5, 2])
        for offset in (4, 7, 10, 11, 20, 23):
            corrupt = list(after)
            corrupt[offset] += 1
            with self.assertRaises(RuntimeError, msg='corrupted plan index ' + str(offset)):
                probe.validate_plan(before, corrupt, rows, 8)


if __name__ == '__main__':
    unittest.main()
