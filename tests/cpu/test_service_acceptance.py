import copy
import unittest

from src.weave.service_acceptance import validate_coverage_delta, compare_first_token, validate_checkpoint_gate


def snapshot():
    return {'rank': 0, 'kernel': 'wmma', 'source_sha256': 'source', 'graph_enabled': True,
            'layers': {str(i): {'device_calls': [1, 3, 0], 'capture_calls': 1,
                              'python_decode_calls': 3} for i in range(43)}}


class ServiceAcceptanceTest(unittest.TestCase):
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

    def test_first_token_comparison_uses_same_prompt_and_token_ids(self):
        a = {'input_sha256': 'same', 'response': {'meta_info': {
            'output_token_logprobs': [[-0.4, 12, None]],
            'output_top_logprobs': [[[-0.4, 12, None], [-2., 9, None]]]}}}
        b = copy.deepcopy(a)
        self.assertEqual(compare_first_token(a, b)['status'], 'passed')
        b['input_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'prompt'):
            compare_first_token(a, b)
        b = copy.deepcopy(a)
        b['response']['meta_info']['output_token_logprobs'][0][1] = 13
        with self.assertRaisesRegex(ValueError, 'token'):
            compare_first_token(a, b)


if __name__ == '__main__':
    unittest.main()
