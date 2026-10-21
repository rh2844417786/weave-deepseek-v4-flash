"""Serving contract tests; all process/network dependencies are replaced on CPU."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tests.cpu.test_gpqa_c32_contract import SummaryTests

SPEC = importlib.util.find_spec('scripts.run_native_budget')
if SPEC:
    from scripts import run_native_budget as controller
else:
    controller = None

IDENTITY = {'native_budget_source_sha256': 'b' * 64, 'native_budget_upstream_sha256': {'marlin.cuh': 'a' * 64},
            'source_sha256': 'c' * 64, 'checkpoint_content_sha256': 'd' * 64, 'image_id': 'sha256:pinned',
            'observer_source_sha256': 'e' * 64, 'graph_sources': {'graph.py': 'f' * 64}}


def budgets(calls, bucket=6):
    return [{'schema_version': 1, 'method': 'native_marlin_dynamic_cta_v1', 'numerical_policy': 'record_only',
             'policy': {'mode': 'adaptive', 'target_percent': 75, 'ep_size': 4, 'min_percent': 25, 'buckets': 8},
             'rank': rank, 'layer_count': 43, 'source_sha256': 'b' * 64, 'upstream_sha256': {'marlin.cuh': 'a' * 64},
             'rows': [{'layer': layer, 'role': role, 'calls': calls, 'empty_calls': calls if bucket == 0 else 0,
                       'native_grid': 128, 'last_active_workers': 96 if bucket else 0,
                       'last_global_blocks': 64, 'last_local_blocks': 16 if bucket else 0,
                       'last_local_routes': 48 if bucket else 0, 'last_total_routes': 192,
                       'last_n_tiles': 16, 'last_k_tiles': 16,
                       'bucket_counts': [calls if i == bucket else 0 for i in range(9)]}
                      for layer in range(43) for role in ('gate_up', 'down')]} for rank in range(4)]


def graphs(calls):
    return [{'rank': rank, 'active': False, 'graph_mode': 'full', 'backend': 'FullCudaGraphBackend',
             'numerical_policy': 'record_only', 'captured_batches': list(range(1, 33)),
             'capture_successes': 32, 'capture_attempts': 32, 'capture_failures': 0, 'failed_replays': 0,
             'source_sha256': IDENTITY['graph_sources'], 'observer_source_sha256': 'e' * 64,
             'model_forwards': {key: 0 for key in ('capture', 'warmup', 'eager_decode', 'other')},
             'successful_replays': calls, 'raw_batch_histogram': {'32': calls}, 'captured_batch_histogram': {'32': calls}}
            for rank in range(4)]


class NativeBudgetControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(controller, 'native budget controller must exist')

    def check(self, before, after):
        return controller.validate_budget_delta(before, after, {'replays': 5}, IDENTITY, target_percent=75)

    def test_every_layer_role_matches_actual_replay_and_zero_variation_is_reported(self):
        result = self.check(budgets(2), budgets(7))
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['total_calls'], 1720)
        self.assertEqual(result['rows_with_budget_variation'], 0)
        self.assertFalse(result['budget_variation_observed'])
        self.assertEqual(result['ranks']['0']['rows'][0]['calls'], 5)
        self.assertEqual(result['numerical_policy'], 'record_only')

    def test_empty_local_launches_remain_valid_covered_replays(self):
        result = self.check(budgets(2, 0), budgets(7, 0))
        self.assertEqual(result['total_empty_calls'], 1720)
        self.assertEqual(result['status'], 'passed')

    def test_missing_duplicate_regressed_and_wrong_source_rows_fail(self):
        for mutation in ('missing', 'duplicate', 'calls', 'histogram', 'source', 'policy', 'nonfinite'):
            before, after = budgets(2), budgets(7)
            row = after[0]['rows'][0]
            if mutation == 'missing': after[0]['rows'].pop()
            if mutation == 'duplicate': after[0]['rows'][-1] = copy.deepcopy(row)
            if mutation == 'calls': row['calls'] = 6
            if mutation == 'histogram': row['bucket_counts'][6] = 6
            if mutation == 'source': after[1]['source_sha256'] = 'x' * 64
            if mutation == 'policy': after[1]['policy']['target_percent'] = 100
            if mutation == 'nonfinite': row['last_active_workers'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check(before, after)

    def test_preview_is_offline_two_native_full_graph_arms(self):
        output = io.StringIO()
        with patch.object(controller.subprocess, 'run', side_effect=AssertionError('preview executed')), contextlib.redirect_stdout(output):
            self.assertEqual(controller.main(['--work-root', '/work', '--records', '/prepared/main.jsonl']), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['seconds_per_arm'], 1200)
        self.assertEqual(set(value['groups']), {'native', 'adaptive'})
        self.assertEqual(value['initial_concurrency'], 32)
        self.assertFalse(value['complete_weave_claim'])
        for mode, command in value['groups'].items():
            self.assertEqual(command[command.index('--backend') + 1], 'native')
            self.assertEqual(command[command.index('--decode-graph') + 1], 'full')
            self.assertIn('--graph-c32', command)
            self.assertNotIn('--native-p0', command)
            self.assertEqual('--native-budget' in command, mode == 'adaptive')

    def test_whole_window_ratio_survives_refill_exhaustion_with_explicit_scope(self):
        native, adaptive = SummaryTests().summary(), SummaryTests().summary()
        adaptive.update(decode_tokens_in_window=576, output_tokens_in_window=608, decode_tokens_per_second=.48,
            requests_started=448, requests_completed=417, requests_unfinished=31, requests_unattempted=0,
            completed_row_indices=list(range(417)), refill_exhausted=True, refill_exhausted_monotonic=1100.)
        result = controller.compare_budget_groups({'native': native, 'adaptive': adaptive})
        self.assertEqual(result['adaptive_vs_native_whole_window_ratio'], 2.)
        self.assertFalse(result['sustained_c32_ratio_available'])
        self.assertIsNone(result['adaptive_vs_native_sustained_c32_ratio'])
        self.assertEqual(result['decode_tokens_per_second'], {'native': .24, 'adaptive': .48})

    def test_public_export_allowlist_excludes_private_content_and_covers_adaptive(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'; (run / 'adaptive').mkdir(parents=True)
            (run / 'experiment.json').write_text('{"status":"failed"}')
            (run / 'adaptive/budget-before.json').write_text(json.dumps(budgets(2)))
            (run / 'adaptive/server-private.log').write_text('secret')
            (run / 'records.jsonl').write_text('secret')
            destination = Path(tmp) / 'public'
            manifest = controller.export_public(run, destination)
            self.assertEqual(set(manifest['files']), {'experiment.json', 'adaptive/budget-before.json'})
            self.assertFalse(manifest['raw_content_exported'])
            self.assertEqual((destination / 'adaptive/budget-before.json').stat().st_mode & 0o777, 0o664)
            (run / 'adaptive/budget-after.json').write_text('{"question":"secret"}')
            with self.assertRaises(ValueError): controller.export_public(run, Path(tmp) / 'bad')

    def test_validation_failure_cleans_owned_processes_and_exports_only_error_type(self):
        from src.weave.gpu_allocation import GPUAllocation
        with tempfile.TemporaryDirectory() as tmp:
            owners = [MagicMock(), MagicMock()]; output = io.StringIO()
            with patch.object(controller, 'OwnedChild', side_effect=owners), patch.object(controller, 'allocated_gpus', return_value=GPUAllocation.parse('0,1,2,3')), patch.object(controller, 'load_gpqa_workload', side_effect=ValueError('private failure')), contextlib.redirect_stdout(output):
                code = controller.main(['--work-root', tmp, '--records', tmp + '/main.jsonl', '--execute'])
            self.assertEqual(code, 1)
            result = json.loads(output.getvalue())
            report = json.loads((Path(result['export_directory']) / 'experiment.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_type'], 'ValueError')
            self.assertNotIn('private failure', json.dumps(report))
            for owner in owners: owner.stop.assert_called_once()

    def test_idle_snapshots_bracket_measurement_without_budget_reads_on_native(self):
        for variant in ('native', 'adaptive'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); out = root / 'experiment'; out.mkdir()
                args = SimpleNamespace(work_root=root, records=root / 'main.jsonl', model_path=root / 'model',
                    port=30001, evidence_port_base=30110, graph_evidence_port_base=30310,
                    native_budget_target_percent=75, setup_timeout=7200., warmup_timeout=960., drain_timeout=60.)
                run_name = 'native-ep-unit-' + variant; folder = out / variant
                state = {'phase': 'starting', 'polls': 0}; snapshots = []
                summary = SummaryTests().summary(); workload = SimpleNamespace(identity=summary['workload'])
                server = MagicMock(); server.poll.return_value = None
                client = MagicMock(); client.returncode = 0
                owner, client_owner = MagicMock(), MagicMock()
                def spawn_server(command, lifecycle, **kwargs):
                    lifecycle.parent.mkdir()
                    controller.atomic_json(lifecycle.parent / 'runtime.json', dict(runtime_status='passed', resolved_image_id='sha256:pinned', kernel_source_sha256='c'*64, host_gpu_allocation='0,1,2,3'))
                    (lifecycle.parent / 'server.log').write_text('')
                    return server
                def spawn_client(command, lifecycle, **kwargs):
                    controller.atomic_json(folder / 'warmup-ready.json', {'ready_monotonic': 100.})
                    controller.atomic_json(folder / 'measurement-summary.json', summary)
                    state['phase'] = 'idle_before'
                    return client
                def poll():
                    state['polls'] += 1
                    if state['polls'] == 3:
                        state['phase'] = 'idle_after'; return 0
                    return None
                def snapshot(base, *, deadline, endpoint):
                    self.assertIn(state['phase'], ('idle_before', 'idle_after'))
                    snapshots.append((state['phase'], endpoint))
                    count = 2 if state['phase'] == 'idle_before' else 7
                    return budgets(count) if endpoint == 'budget-coverage' else graphs(count)
                real_atomic = controller.atomic_json
                def publish(path, value):
                    real_atomic(path, value)
                    if Path(path).name == 'measurement-go.json':
                        state['phase'] = 'measuring'
                        real_atomic(folder / 'load-window.json', {'started_monotonic': 100., 'deadline_monotonic': 1300.})
                owner.spawn.side_effect = spawn_server; client_owner.spawn.side_effect = spawn_client; client.poll.side_effect = poll
                with patch.object(controller, 'wait_ready'), patch.object(controller, 'assert_ports_free'), patch.object(controller, 'flush_cache_when_idle'), patch.object(controller, 'baseline_coverage', side_effect=snapshot), patch.object(controller, 'atomic_json', side_effect=publish), patch.object(controller.time, 'monotonic', return_value=100.), patch.object(controller.time, 'sleep'):
                    result = controller.run_group(args, variant, out, run_name, IDENTITY, workload, 'sha256:pinned', {'CUDA_VISIBLE_DEVICES':'0,1,2,3'}, owner, client_owner, lambda: None)
                self.assertEqual(result['status'], 'passed')
                endpoints = [endpoint for phase, endpoint in snapshots]
                self.assertEqual(endpoints.count('budget-coverage'), 2 if variant == 'adaptive' else 0)
                self.assertEqual(endpoints.count('graph-coverage'), 2)
