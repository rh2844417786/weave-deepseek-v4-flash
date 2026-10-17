"""Fail-closed evidence gates; fixtures contain no private tensors or routes."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from src.weave import hopper_evidence as evidence


def identity():
    capture = {key: '1' * 64 for key in (
        'kernel_source_sha256', 'checkpoint_content_sha256', 'dataset_sha256',
        'capture_source_sha256', 'tokenizer_manifest_sha256')}
    capture.update(capture_id='0ad2a8bd-d12a-4685-95c1-5ab7dcc715ef', image_id='sha256:' + '2' * 64)
    return {'candidate_source_sha256': '3' * 64, 'bf16_source_sha256': '1' * 64,
            'replay_source_sha256': '4' * 64, 'checkpoint_content_sha256': '1' * 64,
            'image_id': capture['image_id'], 'capture_manifest_sha256': '5' * 64,
            'capture_identity': capture, 'warmups': 3, 'repetitions': 20,
            'variants': ['bf16_eager', 'fp8_eager', 'bf16_graph', 'fp8_graph'],
            'layers': list(range(43)), 'samples': [64, 128],
            'invocation_id': '7c9ad43c-ac4f-46eb-ac26-62f26cb60b35'}


def metrics(relative=.001, allclose=True, *, precision=False):
    return {'finite': True, 'relative_l2': relative, 'max_abs': .001,
            'allclose': allclose, 'passed': relative <= .025 and (precision or allclose)}


def checks():
    return {'baseline': metrics(), 'implementation': metrics(), 'precision': metrics(precision=True)}


def rank_report(rank=0):
    cases = []
    for name, batch, mode in (
        ('eager_b1', 1, 'eager'), ('eager_b7', 7, 'eager'), ('eager_b17', 17, 'eager'),
        ('eager_b32', 32, 'eager'), ('eager_shrink_b7', 7, 'eager'),
        ('eager_shrink_b1', 1, 'eager'), ('zero_b32', 32, 'eager'),
        ('graph_update_0', 32, 'graph'), ('graph_update_1', 32, 'graph')):
        cases.append({'case': name, 'batch': batch, 'mode': mode, 'checks': checks(),
                      'metadata_exact': True, 'nonlocal_slots_zero': True})
    points = []
    for layer in range(43):
        for sample in (64, 128):
            for variant in ('bf16_eager', 'fp8_eager', 'bf16_graph', 'fp8_graph'):
                points.append({'layer': layer, 'sample': sample, 'variant': variant,
                    'gpu_ms': [2. if variant.startswith('bf16') else 1.] * 20,
                    'timing_valid': True, 'checks': checks(),
                    'workload': {'batch': 32, 'local_slots': 48, 'active_experts': 12,
                                 'max_rows_per_expert': 4, 'bf16_groups': 12},
                    'stable_addresses': True, 'graph_input_update_verified': True})
    return {'schema_version': 1, 'kind': 'hopper_fp8_local_expert_replay', 'rank': rank,
        'status': 'complete', 'identity': identity(), 'measurement_scope': 'online_group_to_routed_local_fp32_slots',
        'used_for_formal_performance': False, 'service_promotion_allowed': False,
        'replica_sha256': '6' * 64, 'synthetic': {'status': 'passed', 'cases': cases,
            'stable_addresses': True, 'graph_input_updates': 2, 'fp4_patterns': True, 'empty_rank': True},
        'points': points, 'resources': {'bf16': {'group_capacity': 192}, 'fp8': {'threads': 128}},
        'runtime': {'device_name': 'NVIDIA H100 80GB HBM3', 'capability': [9, 0],
                    'multiprocessors': 132, 'device_count': 4, 'cuda_version': '13.0', 'torch_version': '2.11.0'},
        'stop_reason': 'none'}


class HopperEvidenceTest(unittest.TestCase):
    def test_source_identity_binds_oracle_loaders_and_transitive_capture_dependencies(self):
        source_root = Path(__file__).resolve().parents[2]
        paths = ('src/weave/hopper_evidence.py', 'tests/gpu/check_hopper_candidate.py',
                 'src/weave/hopper_reference.py', 'src/weave/hopper_candidate.py', 'src/weave/ops.py',
                 'src/weave/replay_trace.py', 'src/weave/checkpoint_identity.py',
                 'src/weave/weights.py', 'src/weave/staged_plan.py', 'tests/gpu/check_checkpoint.py',
                 'src/weave/calibration.py', 'src/weave/gpu_allocation.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in paths:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / name, target)
            # Only relocate source discovery; hashing and identity validation
            # execute normally over actual copied files, without a fake digest.
            with mock.patch.object(evidence, '__file__', str(root / 'src/weave/hopper_evidence.py')):
                original = evidence.replay_source_sha256()
                expected_identity = identity()
                expected_identity['replay_source_sha256'] = original
                for name in paths:
                    with self.subTest(source=name):
                        target = root / name
                        content = target.read_bytes()
                        try:
                            target.write_bytes(content + b'\n# Changed local source identity.\n')
                            changed = evidence.replay_source_sha256()
                            self.assertNotEqual(changed, original)
                            report = rank_report()
                            report['identity']['replay_source_sha256'] = changed
                            with self.assertRaises(ValueError):
                                evidence.validate_rank_report(report, expected_identity)
                        finally:
                            target.write_bytes(content)

    def test_complete_four_rank_pairing_counts_and_speedups(self):
        reports = [rank_report(rank) for rank in range(4)]
        summary = evidence.aggregate_rank_reports(reports, identity())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['point_count'], 1376)
        self.assertEqual(summary['case_count_per_rank'], 86)
        self.assertTrue(summary['numerical_promotion_eligible'])
        self.assertFalse(summary['service_promotion_allowed'])
        self.assertFalse(summary['used_for_formal_performance'])
        self.assertAlmostEqual(summary['paired_speedups']['eager']['geomean'], 2.)
        self.assertEqual(summary['paired_speedups']['graph']['pairs'], 344)

    def test_complete_rejects_missing_layer_sample_variant_and_duplicate(self):
        for mutate in (lambda r: r['points'].pop(),
                       lambda r: r['points'].append(copy.deepcopy(r['points'][0])),
                       lambda r: r['points'][0].update(sample=0),
                       lambda r: r['points'][0].update(variant='serving_tps')):
            report = rank_report(); mutate(report)
            with self.assertRaises(ValueError): evidence.validate_rank_report(report)

    def test_timings_require_every_repetition_and_matching_workload(self):
        for mutate in (lambda p: p['gpu_ms'].pop(), lambda p: p['gpu_ms'].__setitem__(0, 0.),
                       lambda p: p.update(stable_addresses=False),
                       lambda p: p.update(graph_input_update_verified=False)):
            report = rank_report(); mutate(report['points'][0])
            with self.assertRaises(ValueError): evidence.validate_rank_report(report)
        reports = [rank_report(r) for r in range(4)]
        reports[0]['points'][1]['workload']['local_slots'] = 47
        with self.assertRaises(ValueError): evidence.aggregate_rank_reports(reports)

    def test_pending_synthetic_gate_cannot_claim_valid_real_timing(self):
        report = rank_report()
        report.update(status='partial', stop_reason='budget')
        report['synthetic'] = {'status': 'pending', 'cases': [], 'stable_addresses': False,
                              'graph_input_updates': 0, 'fp4_patterns': False, 'empty_rank': False}
        with self.assertRaises(ValueError): evidence.validate_rank_report(report)

    def test_resource_export_uses_explicit_numeric_allowlist(self):
        self.assertEqual(evidence.public_resources({'threads': 128, 'a_map_shape': (64, 64, 4096),
            'activation_quantization': 'metadata', 'ids': [1, 2], 'opaque': [[.2]]}),
            {'threads': 128, 'a_map_shape': [64, 64, 4096]})

    def test_precision_failure_retains_all_timings_as_diagnostics(self):
        reports = [rank_report(rank) for rank in range(4)]
        reports[2]['points'][17]['checks']['precision'] = metrics(.04, precision=True)
        summary = evidence.aggregate_rank_reports(reports)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['point_count'], 1376)
        self.assertTrue(summary['implementation_passed'])
        self.assertFalse(summary['precision_passed'])
        self.assertFalse(summary['numerical_promotion_eligible'])
        self.assertTrue(summary['timings_diagnostic_only'])
        self.assertEqual(summary['precision_failed_points'], 1)

    def test_implementation_failure_cannot_leave_valid_timing(self):
        report = rank_report()
        report['points'][0]['checks']['implementation'] = metrics(.04)
        with self.assertRaises(ValueError): evidence.validate_rank_report(report)
        report['status'], report['stop_reason'] = 'failed', 'implementation_mismatch'
        report['points'][0].update(timing_valid=False, gpu_ms=[])
        evidence.validate_rank_report(report)
        summary = evidence.aggregate_rank_reports([report] + [rank_report(r) for r in (1, 2, 3)])
        self.assertEqual(summary['status'], 'failed')
        self.assertFalse(summary['implementation_passed'])
        self.assertFalse(summary['numerical_promotion_eligible'])

    def test_claimed_pass_must_match_scalar_errors_and_allclose(self):
        for field, value in (('relative_l2', .5), ('allclose', False), ('finite', False), ('max_abs', float('nan'))):
            report = rank_report(); report['points'][0]['checks']['implementation'][field] = value
            with self.assertRaises(ValueError): evidence.validate_rank_report(report)

    def test_partial_budget_report_is_salvage_only_and_not_resumable(self):
        reports = [rank_report(r) for r in range(4)]
        for report in reports:
            report.update(status='partial', stop_reason='budget')
            report['points'] = report['points'][:7]
        summary = evidence.aggregate_rank_reports(reports)
        self.assertEqual(summary['status'], 'partial')
        self.assertFalse(summary['numerical_promotion_eligible'])
        self.assertTrue(summary['timings_diagnostic_only'])
        with self.assertRaises(ValueError): evidence.validate_rank_report(reports[0], require_complete=True)

    def test_public_schema_rejects_private_payload_at_every_boundary(self):
        mutations = [lambda r: r.update(ids=[[1, 2]]),
            lambda r: r['points'][0].update(routing=[[.1, .2]]),
            lambda r: r['points'][0]['checks'].update(activations=[1.]),
            lambda r: r['resources']['fp8'].update(expert_slots=[1, 2]),
            lambda r: r['resources']['fp8'].update(unknown_array=[[1, 2]]),
            lambda r: r.update(measurement_scope='serving'),
            lambda r: r.update(used_for_formal_performance=True)]
        for mutate in mutations:
            report = rank_report(); mutate(report)
            with self.assertRaises(ValueError): evidence.validate_rank_report(report)

    def test_replica_identity_invocation_and_synthetic_gate_must_match(self):
        for mutate in (lambda r: r.update(replica_sha256='7'*64),
                       lambda r: r['identity'].update(candidate_source_sha256='8'*64),
                       lambda r: r['identity'].update(invocation_id='ce597416-0cd6-416b-a205-27be29ab7c64'),
                       lambda r: r['synthetic'].update(graph_input_updates=1),
                       lambda r: r['synthetic']['cases'].pop(),
                       lambda r: r['runtime'].update(capability=[8, 0])):
            reports = [rank_report(r) for r in range(4)]; mutate(reports[3])
            with self.assertRaises(ValueError): evidence.aggregate_rank_reports(reports)

    def test_archived_capture_source_is_separate_from_current_baseline(self):
        value = identity()
        value['bf16_source_sha256'] = 'a' * 64
        value['capture_identity']['capture_source_sha256'] = 'b' * 64
        evidence.validate_identity(value)

    def test_current_checkpoint_and_capture_identity_cannot_disagree(self):
        value = identity(); value['checkpoint_content_sha256'] = 'a' * 64
        with self.assertRaises(ValueError): evidence.validate_identity(value)
        value = identity(); value['warmups'] = 0
        with self.assertRaises(ValueError): evidence.validate_identity(value)

    def test_atomic_report_is_readable_and_0664_and_refuses_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rank.json'
            evidence.atomic_report(path, {'status': 'partial'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'partial'})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o664)
            with self.assertRaises(ValueError): evidence.atomic_report(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'status': 'partial'})
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_gpu_cli_imports_without_torch_and_requires_fresh_invocation(self):
        path = Path(__file__).parents[1] / 'gpu/check_hopper_candidate.py'
        spec = importlib.util.spec_from_file_location('hopper_gate_cli', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = module.parse_args(['--model-path', '/model', '--capture-dir', '/capture',
            '--output-dir', '/output', '--identity-file', '/identity.json',
            '--invocation-id', identity()['invocation_id']])
        self.assertEqual((args.warmups, args.repetitions, args.budget_seconds), (3, 20, 7200))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.parse_args(['--resume'])


if __name__ == '__main__':
    unittest.main()
