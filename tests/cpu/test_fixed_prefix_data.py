"""Behavioral controls for the bounded fixed-prefix computation probe."""
import asyncio
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def workload():
    return SimpleNamespace(records=tuple({
        'row_index': 447-i, 'prompt_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
        'input_tokens': 2, 'input_token_ids': [i, 2]} for i in range(448)),
        identity={'record_sha256': 'a'*64, 'order_seed': 42})


def module():
    spec = importlib.util.find_spec('src.weave.fixed_prefix_data')
    if spec is None:
        raise AssertionError('fixed-prefix data and client helpers not implemented')
    return importlib.import_module('src.weave.fixed_prefix_data')


class Response:
    status = 200

    def __init__(self, payload=None, kind='ok'):
        self.payload, self.kind = payload, kind

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        if self.kind in ('stall', 'abort_fail'):
            await asyncio.sleep(5)
        if self.kind == 'transport':
            raise RuntimeError('PRIVATE_TRANSPORT_CONTENT')
        rows = []
        for index, rid in enumerate(self.payload['rid']):
            count = self.payload['sampling_params'][index]['max_new_tokens']
            rows.append({'output_ids': [index+7]*count,
                         'text': 'PRIVATE_TEXT',
                         'meta_info': {'id': rid, 'completion_tokens': count,
                                       'finish_reason': {'type': 'length', 'length': count}}})
        if self.kind == 'foreign':
            rows[0]['meta_info']['id'] = 'foreign'
        if self.kind == 'duplicate':
            rows[0]['meta_info']['id'] = rows[1]['meta_info']['id']
        if self.kind == 'missing_ids':
            rows[0].pop('output_ids')
        if self.kind == 'wrong_count':
            rows[0]['meta_info']['completion_tokens'] -= 1
        if self.kind == 'stop':
            rows[0]['meta_info']['finish_reason'] = {'type': 'stop', 'matched': 'PRIVATE_STOP'}
        if self.kind == 'nonfinite':
            rows[0]['meta_info']['latency'] = float('nan')
        if self.kind == 'changed':
            rows[0]['output_ids'][0] += 1
        return list(reversed(rows))


class Session:
    def __init__(self, kind='ok'):
        self.kind, self.payloads, self.aborted = kind, [], []

    def post(self, url, json, **kwargs):
        if url.endswith('/abort_request'):
            self.aborted.append(json['rid'])
            response = Response()
            response.status = 500 if self.kind == 'abort_fail' else 200
            return response
        if not url.endswith('/generate'):
            raise AssertionError('unexpected endpoint')
        self.payloads.append(json)
        return Response(json, self.kind)

    def get(self, *args, **kwargs):
        raise AssertionError('controller owns cache flush and warmup')


class FixedPrefixDataTests(unittest.IsolatedAsyncioTestCase):
    def test_config_tracks_ordered_token_commitments_and_requires_full_source(self):
        data = module()
        source = workload()
        config = data.prepare_config(source, steps=8, vocab_size=1024)
        self.assertEqual(config['requests'][0], {
            'row_slot': 0, 'original_row_index': 447, 'input_tokens': 2,
            'prompt_sha256': hashlib.sha256(b'0').hexdigest(),
            'input_ids_sha256': digest([0, 2])})
        self.assertEqual(len(config['requests']), 32)
        self.assertEqual(config['steps'], 8)
        from src.weave.native_p0_pair_client import selection_identity
        self.assertEqual(config['selection_sha256'], selection_identity(source)['selection_sha256'])
        source.records[40]['input_token_ids'][0] = 99
        self.assertEqual(data.prepare_config(source, 8, 1024), config)
        with self.assertRaisesRegex(ValueError, '448'):
            data.prepare_config(SimpleNamespace(records=source.records[:32]), 8, 1024)
        with self.assertRaisesRegex(ValueError, 'vocab'):
            data.prepare_config(source, 8, 16)

    async def run_client(self, directory, phase='seed', kind='ok', steps=8, timeout=.2):
        data = module()
        session = Session(kind)
        result = await data.run_phase(
            session, 'http://127.0.0.1:30000', workload(), Path(directory)/('client-'+phase),
            Path(directory)/'private'/'teacher.json', phase=phase, steps=steps,
            timeout_seconds=timeout, processor='SERIALIZED_PROCESSOR')
        self.assertEqual(result, json.loads((Path(directory)/('client-'+phase)/
                                            'fixed-prefix-client-summary.json').read_text()))
        return result, session

    async def test_one_atomic_batch_reorders_response_ids_and_saves_private_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, session = await self.run_client(tmp)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['requests_completed'], 32)
            self.assertTrue(result['cleanup_complete'])
            self.assertEqual(len(session.payloads), 1)
            payload = session.payloads[0]
            self.assertEqual(len(payload['input_ids']), 32)
            self.assertEqual(len(set(payload['rid'])), 32)
            self.assertFalse(payload['stream'])
            self.assertFalse(payload['return_logprob'])
            self.assertEqual(payload['custom_logit_processor'], 'SERIALIZED_PROCESSOR')
            for index, params in enumerate(payload['sampling_params']):
                self.assertEqual(params['max_new_tokens'], 8)
                self.assertIs(params['ignore_eos'], True)
                self.assertEqual(params['temperature'], 0)
                self.assertEqual(params['custom_params']['weave_fixed_prefix']['row_slot'], index)
            teacher_path = Path(tmp)/'private'/'teacher.json'
            teacher = json.loads(teacher_path.read_text())
            self.assertEqual(teacher['output_ids'], [[i+7]*8 for i in range(32)])
            self.assertEqual(teacher['sha256'], digest(teacher['output_ids']))
            self.assertEqual(teacher_path.stat().st_mode & 0o777, 0o664)
            public = json.dumps(result)
            for secret in ('PRIVATE_TEXT', '"output_ids"', '"input_ids"', '"rid"'):
                self.assertNotIn(secret, public)
            for rid in payload['rid']:
                self.assertNotIn(rid, public)

    async def test_measure_and_two_step_trace_require_the_prescribed_input_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            await self.run_client(tmp)
            measured, _ = await self.run_client(tmp, phase='measure')
            traced, _ = await self.run_client(tmp, phase='trace', steps=2)
            self.assertEqual(measured['status'], 'passed')
            self.assertEqual(traced['status'], 'passed')
            self.assertEqual(traced['output_tokens'], 64)
            self.assertEqual(traced['teacher_sha256'], measured['teacher_sha256'])
            changed, _ = await self.run_client(tmp, phase='measure', kind='changed')
            self.assertEqual(changed['status'], 'failed')
            self.assertIn('forced_continuation_integrity_mismatch', changed['error_codes'])
            self.assertEqual(changed['numerical_policy'], 'record_only')

    async def test_invalid_response_content_is_sanitized_and_all_owned_requests_abort(self):
        for kind, code in (
            ('foreign', 'response_request_id_mismatch'), ('duplicate', 'duplicate_response_request_id'),
            ('missing_ids', 'actual_output_ids_required'), ('wrong_count', 'output_token_count_mismatch'),
            ('stop', 'length_finish_required'), ('nonfinite', 'response_nonfinite_value'),
            ('transport', 'client_transport_error'), ('stall', 'fixed_prefix_timeout')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                result, session = await self.run_client(tmp, kind=kind, timeout=.01)
                self.assertEqual(result['status'], 'failed')
                self.assertIn(code, result['error_codes'])
                self.assertEqual(len(session.aborted), 32)
                self.assertTrue(result['cleanup_complete'])
                self.assertNotIn('PRIVATE', json.dumps(result))
                self.assertFalse((Path(tmp)/'private'/'teacher.json').exists())

    async def test_missing_or_tampered_teacher_fails_before_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing, session = await self.run_client(tmp, phase='measure')
            self.assertEqual(missing['status'], 'failed')
            self.assertFalse(session.payloads)
            await self.run_client(tmp)
            path = Path(tmp)/'private'/'teacher.json'
            teacher = json.loads(path.read_text())
            teacher['output_ids'][0][0] = 500
            path.write_text(json.dumps(teacher))
            bad, session = await self.run_client(tmp, phase='trace', steps=2)
            self.assertIn('teacher_digest_mismatch', bad['error_codes'])
            self.assertFalse(session.payloads)

    async def test_cleanup_failure_and_cancellation_preserve_failure_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad, _ = await self.run_client(tmp, kind='abort_fail', timeout=.01)
            self.assertFalse(bad['cleanup_complete'])
            self.assertIn('abort_failed', bad['error_codes'])
        with tempfile.TemporaryDirectory() as tmp:
            session = Session('stall')
            task = asyncio.create_task(module().run_phase(
                session, 'http://localhost:30000', workload(), Path(tmp)/'public',
                Path(tmp)/'private'/'teacher.json', phase='seed', processor='P'))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            result = json.loads((Path(tmp)/'public'/'fixed-prefix-client-summary.json').read_text())
            self.assertIn('client_cancelled', result['error_codes'])
            self.assertEqual(len(session.aborted), 32)


class FixedPrefixCLITests(unittest.TestCase):
    def test_preview_pinned_source_without_network_or_runtime_import(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.bench_fixed_prefix'),
                             'fixed-prefix CLI missing')
        from scripts import bench_fixed_prefix as cli
        records = Path(__file__).resolve().parents[2]/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl'
        if not records.exists():
            self.skipTest('pinned local corpus unavailable')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            destination = Path(tmp)/'preview'
            with patch.object(cli, 'execute', side_effect=AssertionError('preview made request')):
                code = cli.main(['--records', str(records), '--output-dir', str(destination),
                                 '--base-url', 'http://127.0.0.1:30000', '--phase', 'seed',
                                 '--teacher', str(Path(tmp)/'private'/'teacher.json')])
            value = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(value['status'], 'preview_not_executed')
            self.assertEqual(value['selection']['selection_sha256'],
                             'a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018')
            self.assertEqual(value['input_tokens'], 6059)
            self.assertFalse(destination.exists())


def comparator():
    if importlib.util.find_spec('scripts.compare_fixed_prefix') is None:
        raise AssertionError('fixed-prefix comparator not implemented')
    return importlib.import_module('scripts.compare_fixed_prefix')


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None,
                     'NumPy unavailable in this interpreter; use bundled Python or existing SGLang image')
class FixedPrefixComparisonTests(unittest.TestCase):
    def test_finite_arithmetic_and_top1_differences_are_recorded_without_gate(self):
        compare = comparator()
        import numpy as np
        reference = np.array([[3., 0., 1.], [0., 4., 0.]])
        candidate = np.array([[0., 5., 1.], [8., 0., 0.]])
        result = compare.logit_metrics(reference, candidate)
        self.assertEqual(result['top1_matched'], 0)
        self.assertEqual(result['top1_mismatched'], 2)
        self.assertEqual(result['max_abs'], 8.)
        self.assertAlmostEqual(result['relative_l2'], (114/26)**.5)
        self.assertEqual(result['mean_abs'], 20/6)
        self.assertEqual(result['reference_top1_sha256'], digest([0, 1]))
        self.assertNotIn('passed', result)
        self.assertNotIn('failed', result)
        self.assertNotIn('output_ids', json.dumps(result))

    def test_zero_reference_records_undefined_relative_norm_without_rejecting(self):
        compare = comparator()
        import numpy as np
        same = compare.logit_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
        changed = compare.logit_metrics(np.zeros((2, 3)), np.ones((2, 3)))
        self.assertEqual(same['relative_l2'], 0.)
        self.assertIsNone(changed['relative_l2'])
        self.assertEqual(changed['relative_l2_status'], 'undefined_zero_reference_norm')
        self.assertEqual(changed['max_abs'], 1.)
        self.assertIsNone(changed['cosine_similarity'])
        json.dumps(changed, allow_nan=False)

    def test_malformed_or_nonfinite_logits_remain_hard_failures(self):
        compare = comparator()
        import numpy as np
        for value in (np.array([[float('nan'), 0.]]), np.array([[float('inf'), 0.]])):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                compare.logit_metrics(np.ones((1, 2)), value)
        with self.assertRaisesRegex(ValueError, 'shape'):
            compare.logit_metrics(np.ones((2, 3)), np.ones((3, 2)))

    def test_first_difference_uses_observed_position_separating_prefill_and_decode(self):
        compare = comparator()
        records = [
            {'rank': 0, 'step': 0, 'mode': 'extend', 'max_abs': 0., 'top1_mismatched': 0},
            {'rank': 1, 'step': 1, 'mode': 'decode', 'max_abs': 1e-20, 'top1_mismatched': 0},
            {'rank': 0, 'step': 2, 'mode': 'decode', 'max_abs': 1., 'top1_mismatched': 2}]
        result = compare.difference_summary(records)
        self.assertEqual(result['first_observed_arithmetic_difference']['step'], 1)
        self.assertEqual(result['first_observed_top1_difference']['step'], 2)
        self.assertEqual(result['first_observed_arithmetic_difference']['mode'], 'decode')

    def make_evidence(self, directory):
        compare = comparator()
        import numpy as np
        root = Path(directory)
        config = module().prepare_config(workload(), 2, 512)
        teacher = {'schema_version': 1, 'selection_sha256': config['selection_sha256'],
                   'steps': 2, 'output_ids': [[slot+7]*2 for slot in range(32)]}
        teacher['sha256'] = digest(teacher['output_ids'])
        (root/'private').mkdir()
        (root/'private'/'teacher.json').write_text(json.dumps(teacher))
        arrays = {}
        for arm in compare.ARMS:
            folder = root/arm/'private'
            (folder/'measure').mkdir(parents=True)
            (folder/'config.json').write_text(json.dumps(config))
            for rank in compare.RANKS:
                records = []
                for step in range(2):
                    order = list(reversed(range(32))) if arm == 'native_b' else list(range(32))
                    values = np.zeros((32, 512))
                    values[:, 0] = 1.
                    if arm == 'p0_full' and step == 1:
                        values[:, 1] = 10.
                    path = folder/'measure'/f'rank{rank}-step{step:03d}.pt'
                    path.write_bytes(b'fixture-serialized-tensor')
                    arrays[path] = values
                    seq_lens = [2+step]*32
                    records.append({'step': step, 'batch_size': 32,
                        'mode': 'extend' if step == 0 else 'decode', 'row_slot_order': order,
                        'seq_lens': seq_lens, 'positions': [n-1 for n in seq_lens],
                        'original_logits_finite': True, 'logits_tensor_sha256': 'b'*64,
                        'logits_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'input_ids_sha256': digest([teacher['output_ids'][slot][step-1] for slot in order]) if step else 'd'*64,
                        'input_ids_by_row_sha256': [digest([teacher['output_ids'][slot][step-1]]) if step else
                                                    config['requests'][slot]['input_ids_sha256'] for slot in order],
                        'req_pool_indices_sha256': 'e'*64, 'out_cache_loc_sha256': 'f'*64,
                        'captured': False,
                        'sampled_ids_sha256': digest([teacher['output_ids'][slot][step] for slot in order]),
                        'native_argmax_sha256': digest(values.argmax(axis=1).tolist())})
                proof = {'schema_version': 1, 'status': 'passed', 'phase': 'measure', 'rank': rank,
                    'world_size': 4, 'selection_sha256': config['selection_sha256'],
                    'config_sha256': digest(config),
                    'teacher_sha256': teacher['sha256'], 'expected_steps': 2, 'completed_steps': [0, 1],
                    'records': records, 'fixed_prefix_source_sha256': 'a'*64,
                    'native_sources': {'sampling.py': 'b'*64}, 'image_id': 'sha256:'+'c'*64,
                    'sampler_source_closure_sha256': digest({'sampling.py': 'b'*64}),
                    'numerical_policy': 'record_only'}
                (folder/f'measure-rank{rank}.json').write_text(json.dumps(proof))
        return root, config, teacher, arrays

    def test_complete_five_arm_report_records_all_mismatch_and_no_raw_data(self):
        compare = comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root, config, teacher, arrays = self.make_evidence(tmp)
            with patch.object(compare, 'load_snapshot', side_effect=lambda path, *args: arrays[path]):
                result = compare.compare_run(root)
            self.assertEqual(result['status'], 'comparison_complete')
            self.assertEqual([row['candidate_arm'] for row in result['comparisons']],
                             ['native_b', 'p0_full', 'p0_events', 'p0_buffers'])
            self.assertEqual(result['comparisons'][0]['top1_mismatched'], 0)
            changed = result['comparisons'][1]
            self.assertEqual(changed['status'], 'comparison_complete')
            self.assertEqual(changed['top1_mismatched'], 128)
            self.assertEqual(changed['first_observed_arithmetic_difference']['step'], 1)
            self.assertEqual(changed['first_observed_top1_difference']['step'], 1)
            public = json.dumps(result, allow_nan=False)
            for key in ('"output_ids"', '"input_ids"', '"logits"'):
                self.assertNotIn(key, public)

    def test_missing_tensor_source_teacher_or_position_failure_precedes_comparison(self):
        compare = comparator()
        changes = (
            ('source', lambda root, proof: proof['native_sources'].update({'sampling.py': 'f'*64})),
            ('teacher', lambda root, proof: proof.update(teacher_sha256='f'*64)),
            ('position', lambda root, proof: proof['records'][0]['positions'].__setitem__(0, 200)),
            ('input', lambda root, proof: proof['records'][1]['input_ids_by_row_sha256'].__setitem__(0, 'f'*64)),
            ('coverage', lambda root, proof: proof.update(completed_steps=[0])),
            ('sampled', lambda root, proof: proof['records'][1].update(sampled_ids_sha256='f'*64)),
            ('config', lambda root, proof: proof.update(config_sha256='f'*64)),
            ('captured', lambda root, proof: proof['records'][0].update(captured=True)),
            ('file', lambda root, proof: (root/'p0_full/private/measure/rank0-step001.pt').unlink()),
        )
        for name, mutate in changes:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, config, teacher, arrays = self.make_evidence(tmp)
                path = root/'p0_full/private/measure-rank0.json'
                proof = json.loads(path.read_text())
                mutate(root, proof)
                path.write_text(json.dumps(proof))
                with patch.object(compare, 'load_snapshot', side_effect=AssertionError('loaded invalid evidence')):
                    with self.assertRaises(ValueError):
                        compare.compare_run(root)

    def test_cpu_safe_tensor_loader_checks_bytes_shape_and_restores_row_order(self):
        compare = comparator()
        import numpy as np
        from src.weave import fixed_prefix_runtime

        class Tensor:
            device = SimpleNamespace(type='cpu')
            def __init__(self, values):
                self.values, self.shape = values, values.shape
            def is_floating_point(self): return True
            def detach(self): return self
            def double(self): return self
            def numpy(self): return self.values

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.pt'
            path.write_bytes(b'fixture')
            values = np.array([[float(slot), -1.] for slot in reversed(range(32))])
            record = {'logits_file_sha256': hashlib.sha256(b'fixture').hexdigest(),
                      'logits_tensor_sha256': 'a'*64, 'row_slot_order': list(reversed(range(32))),
                      'native_argmax_sha256': digest([0]*32)}
            calls = []
            def loader(filename, **kwargs):
                calls.append(kwargs)
                return Tensor(values)
            fake = SimpleNamespace(load=loader, is_tensor=lambda item: isinstance(item, Tensor))
            with patch.object(fixed_prefix_runtime, 'tensor_sha256', return_value='a'*64):
                result = compare.load_snapshot(path, record, 2, torch_module=fake)
                self.assertEqual(result[:, 0].tolist(), list(range(32)))
                self.assertEqual(calls, [{'map_location': 'cpu', 'weights_only': True}])
                with self.assertRaisesRegex(ValueError, 'shape'):
                    compare.load_snapshot(path, record, 3, torch_module=fake)
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'file_digest'):
                    compare.load_snapshot(path, record, 2, torch_module=fake)


if __name__ == '__main__':
    unittest.main()
