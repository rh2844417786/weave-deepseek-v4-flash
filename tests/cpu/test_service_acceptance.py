import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.weave.service_acceptance import validate_coverage_delta, compare_first_token, validate_checkpoint_gate
from scripts.check_sglang_service import check_service, ROOT


def snapshot():
    return {'rank': 0, 'kernel': 'wmma', 'source_sha256': 'source', 'graph_enabled': True,
            'layers': {str(i): {'device_calls': [1, 3, 0], 'capture_calls': 1,
                              'python_decode_calls': 3} for i in range(43)}}


class ServiceAcceptanceTest(unittest.TestCase):
    def test_logprob_drift_is_recorded_without_failing_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = root/'main.jsonl'
            records.write_text(''.join(json.dumps({'row_index': i, 'input_token_ids': [1,2,3]})+'\n'
                                       for i in range(448)))
            pinned = json.loads((ROOT/'data/gpqa/deepseek_model_manifest.json').read_text())
            (root/'manifest.json').write_text(json.dumps({
                'record_sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
                'count': 448, 'reasoning_mode': 'Think High', 'max_new_tokens': 32768,
                'normal_eos_allowed': True, 'model_revision': pinned['revision'],
                'pinned_model_metadata_sha256': pinned['sha256']}))
            response = {'output_ids': [12]*8, 'meta_info': {
                'output_token_logprobs': [[-.4,12,None]],
                'output_top_logprobs': [[[-.4,12,None],[-2.,9,None]]]}}
            native = root/'native.json'
            native.write_text(json.dumps({'input_sha256': hashlib.sha256(b'[1,2,3]').hexdigest(),
                                          'response': response}))
            changed = copy.deepcopy(response)
            changed['meta_info']['output_top_logprobs'][0][1][1] = 10
            changed['meta_info']['output_token_logprobs'][0][0] = -1.4
            changed['meta_info']['output_top_logprobs'][0][0][0] = -1.4
            output = root/'request.json'
            # Only the external HTTP boundary is replaced; manifest, comparison,
            # exception handling and evidence-file writes all execute normally.
            with patch('scripts.check_sglang_service.http_json', side_effect=[None, changed]):
                result = check_service('http://127.0.0.1:1', records, output, native=native)
            self.assertEqual(result['status'], 'passed')
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence['status'], 'passed')
            self.assertEqual(evidence['response'], changed)
            diagnostic = evidence.get('native_comparison', {})
            self.assertEqual(diagnostic.get('native_only_token_ids'), [9])
            self.assertEqual(diagnostic.get('weave_only_token_ids'), [10])
            self.assertEqual(diagnostic.get('matched_generated_tokens'), 8)
            self.assertEqual(diagnostic['status'], 'recorded')
            self.assertFalse(diagnostic['used_for_acceptance'])
            self.assertAlmostEqual(diagnostic['first_token_logprob_delta'], 1.)
            # Missing optional logprob output must also be a record, not a gate.
            missing = {'output_ids': [12]*8, 'meta_info': {}}
            with patch('scripts.check_sglang_service.http_json', side_effect=[None, missing]):
                result = check_service('http://127.0.0.1:1', records, output, native=native)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['native_comparison']['status'], 'unavailable')
            # Actual generated-token disagreement remains an acceptance failure.
            different = copy.deepcopy(changed)
            different['output_ids'][-1] = 13
            with patch('scripts.check_sglang_service.http_json', side_effect=[None, different]):
                with self.assertRaisesRegex(ValueError, 'generated token IDs differ'):
                    check_service('http://127.0.0.1:1', records, output, native=native)
            self.assertEqual(json.loads(output.read_text())['status'], 'failed')

    def test_changed_topk_support_does_not_hide_common_tail_drift(self):
        # Transcribed numeric table in the server note, not raw run acceptance.
        native_top = [[0.,2581],[-13.375,10318],[-14.25,671],[-14.75,5718],[-15.875,1350],
                      [-17.125,201],[-17.875,11428],[-18.125,666],[-18.625,9832],[-18.625,2337]]
        weave_top = [[0.,2581],[-13.125,10318],[-13.875,671],[-14.875,5718],[-15.625,1350],
                     [-16.625,201],[-17.75,11428],[-18.25,92341],[-18.25,666],[-18.625,1124]]
        def response(top):
            return {'input_sha256': 'same', 'response': {'meta_info': {
                'output_token_logprobs': [[0.,2581,None]], 'output_top_logprobs': [top]}}}
        diagnostic = compare_first_token(response(native_top), response(weave_top))
        self.assertEqual(diagnostic['status'], 'recorded')
        self.assertFalse(diagnostic['used_for_acceptance'])
        self.assertEqual(diagnostic.get('native_only_token_ids'), [2337,9832])
        self.assertEqual(diagnostic.get('weave_only_token_ids'), [1124,92341])
        self.assertEqual(diagnostic.get('max_abs_common_logprob_delta'), .5)
        self.assertEqual(len(diagnostic['common_logprob_deltas']), 8)
        self.assertNotIn('atol', diagnostic)
        self.assertNotIn('failures', diagnostic)

    def test_checkpoint_gate_rejects_old_or_partial_evidence(self):
        runtime = {'runtime_status': 'passed', 'checkpoint_validation_status': 'passed',
                   'child_exit_code': 0, 'resolved_image_id': 'sha256:image', 'kernel_source_sha256': 'new'}
        rank = {'status': 'checkpoint_checks_passed', 'rank': 0, 'layers': list(range(43)),
                'kernels': ['scalar', 'wmma'], 'architecture': [4096, 2048, 256, 6, 43],
                'source_sha256': 'new', 'checkpoint_index_sha256': 'model', 'cases': [
                    {'layer': layer, 'kernel': kernel, 'case': case, 'completed_stolen': 1,
                     'metrics': {'relative_l2': .001}} for layer in [-1]+list(range(43))
                    for kernel in ['scalar', 'wmma']
                    for case in ['remote_K2', 'steal_K2', 'graph_replay0', 'graph_replay1']]}
        rank['checkpoint_content_sha256'] = 'content'
        ranks = [dict(copy.deepcopy(rank), rank=i) for i in range(4)]
        self.assertEqual(validate_checkpoint_gate(runtime, ranks, source_sha256='new', index_sha256='model', checkpoint_sha256='content')['status'], 'passed')
        with self.assertRaisesRegex(ValueError, 'stale'):
            validate_checkpoint_gate(runtime, ranks, source_sha256='old', index_sha256='model', checkpoint_sha256='content')
        with self.assertRaisesRegex(ValueError, 'stale'):
            validate_checkpoint_gate(runtime, ranks, source_sha256='new', index_sha256='model', checkpoint_sha256='different')
        ranks[0]['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_checkpoint_gate(runtime, ranks, source_sha256='new', index_sha256='model', checkpoint_sha256='content')

    def test_device_replays_are_required_not_python_capture_counts(self):
        before = [dict(snapshot(), rank=i) for i in range(4)]
        after = copy.deepcopy(before)
        for rank in after:
            for layer in rank['layers'].values():
                layer['device_calls'][0] += 1
                layer['device_calls'][1] += 4
        result = validate_coverage_delta(before, after, require_graph=True)
        self.assertEqual(result['status'], 'passed')
        for rank in after:
            for layer in rank['layers'].values():
                layer['device_calls'][1] = 3
                layer['capture_calls'] += 4
        with self.assertRaisesRegex(ValueError, 'decode'):
            validate_coverage_delta(before, after, require_graph=True)

    def test_missing_rank_layer_or_eager_decode_cannot_pass_graph(self):
        before = [dict(snapshot(), rank=i) for i in range(4)]
        after = copy.deepcopy(before)
        with self.assertRaises(ValueError):
            validate_coverage_delta(before, after[:3], require_graph=True)
        for rank in after:
            for layer in rank['layers'].values():
                layer['device_calls'][0] += 1
                layer['device_calls'][1] += 2
                layer['python_decode_calls'] += 2
        with self.assertRaisesRegex(ValueError, 'replay'):
            validate_coverage_delta(before, after, require_graph=True)
        del after[0]['layers']['42']
        with self.assertRaisesRegex(ValueError, '43'):
            validate_coverage_delta(before, after, require_graph=False)

    def test_logprob_diagnostics_require_same_prompt_and_record_metadata_tokens(self):
        a = {'input_sha256': 'same', 'response': {'meta_info': {
            'output_token_logprobs': [[-0.4, 12, None]],
            'output_top_logprobs': [[[-0.4, 12, None], [-2., 9, None]]]}}}
        b = copy.deepcopy(a)
        self.assertEqual(compare_first_token(a, b)['status'], 'recorded')
        b['input_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'prompt'):
            compare_first_token(a, b)
        b = copy.deepcopy(a)
        b['response']['meta_info']['output_token_logprobs'][0][1] = 13
        result = compare_first_token(a, b)
        self.assertEqual(result['status'], 'recorded')
        self.assertFalse(result['first_token_matches'])

    def test_invalid_logprobs_are_recorded_as_unavailable(self):
        for value in (float('nan'), float('inf'), .5):
            with self.subTest(value=value):
                item = {'input_sha256': 'same', 'response': {'meta_info': {
                    'output_token_logprobs': [[value, 12, None]],
                    'output_top_logprobs': [[[value, 12, None]]]}}}
                result = compare_first_token(item, copy.deepcopy(item))
                self.assertEqual(result['status'], 'unavailable')
                self.assertFalse(result['used_for_acceptance'])
                self.assertIn('error', result)
                json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
