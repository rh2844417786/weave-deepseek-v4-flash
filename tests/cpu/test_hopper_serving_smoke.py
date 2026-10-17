"""CPU checks for the GPU smoke's independent deterministic fixtures."""
from contextlib import redirect_stderr
import importlib
import io
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def smoke():
    try:
        return importlib.import_module('tests.gpu.check_hopper_serving')
    except ModuleNotFoundError as exc:
        raise AssertionError('four-rank Hopper serving smoke is missing') from exc


class AssemblyFixtureTests(unittest.TestCase):
    def test_cancellation_fixture_distinguishes_reassociated_fp32(self):
        module = smoke()
        f32 = lambda n: struct.unpack('f', struct.pack('f', n))[0]
        different = False
        for slots, expected in module.ORDERED_FIXTURES:
            result = 0.0
            for slot in slots:
                result = f32(result + slot)
            self.assertEqual(result, expected)
            different |= result != sum(slots)
        self.assertTrue(different, 'fixture must reject accumulation in higher precision')

    def test_origin_rows_partition_small_and_uneven_batches(self):
        module = smoke()
        for batch in (1, 7, 17, 32):
            rows = [module.origin_rows(batch, rank) for rank in range(4)]
            self.assertEqual([r for part in rows for r in part], list(range(batch)))
        self.assertEqual(module.origin_rows(1, 3), [])

    def test_routes_include_empty_local_ranks_and_distinct_slots(self):
        module = smoke()
        for variant, (_, batch) in enumerate(module.CASES):
            rows = module.route_rows(batch, variant)
            self.assertEqual(len(rows), batch)
            self.assertTrue(all(len(set(row)) == 6 for row in rows))
            self.assertTrue(all(0 <= item < 32 for row in rows for item in row))
        owners = {expert // 8 for row in module.route_rows(7, 1) for expert in row}
        self.assertEqual(owners, {1})

    def test_agreed_logs_original_exception_only_to_private_stderr(self):
        module = smoke()
        states = []

        def gather(result, state):
            states.append(state)
            result[:] = [state] * 4

        def compile_failure():
            raise ValueError('private compiler detail tensor=[secret]')

        stderr = io.StringIO()
        dist = SimpleNamespace(all_gather_object=gather)
        with patch.object(module, 'dist', dist, create=True), patch.dict(module.os.environ, LOCAL_RANK='2'):
            with redirect_stderr(stderr), self.assertRaises(module.AgreedFailure) as caught:
                module.agreed('compile', compile_failure)
        self.assertIn('rank=2 stage=compile', stderr.getvalue())
        self.assertIn('Traceback (most recent call last)', stderr.getvalue())
        self.assertIn('ValueError: private compiler detail tensor=[secret]', stderr.getvalue())
        self.assertEqual(states, [('compile', 'ValueError')])
        self.assertEqual(str(caught.exception), 'compile')
        self.assertNotIn('secret', json.dumps(states) + str(caught.exception))

    def test_runtime_logs_original_exception_and_preserves_collective_order(self):
        module = smoke()
        states = []

        def gather(result, state):
            states.append(state)
            result[:] = [state] * 4

        def runtime_failure(*args):
            raise RuntimeError('private Triton runtime diagnostic secret')

        original_reduce = lambda *args, **kwargs: None
        dist = SimpleNamespace(all_reduce=original_reduce, all_gather_object=gather)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        stderr = io.StringIO()
        with patch.object(module, 'dist', dist, create=True), patch.object(module, 'torch', torch, create=True):
            with patch.dict(module.os.environ, LOCAL_RANK='3'), redirect_stderr(stderr):
                with self.assertRaises(module.AgreedFailure) as caught:
                    module.run_captured(SimpleNamespace(run=runtime_failure), (), None, 'case')
        self.assertIn('rank=3 stage=case:runtime', stderr.getvalue())
        self.assertIn('RuntimeError: private Triton runtime diagnostic secret', stderr.getvalue())
        self.assertEqual([state[0] for state in states], ['case:pre_collective', 'case:runtime_complete'])
        self.assertTrue(all(state[1] == 'AssertionError' for state in states))
        self.assertEqual(str(caught.exception), 'case:runtime_complete')
        self.assertNotIn('secret', json.dumps(states) + str(caught.exception))
        self.assertIs(dist.all_reduce, original_reduce)

    def test_agreed_failure_does_not_duplicate_private_traceback(self):
        module = smoke()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            try:
                raise module.AgreedFailure('compile')
            except module.AgreedFailure as exc:
                module.private_exception('main', exc)
        self.assertEqual(stderr.getvalue(), '')

    def test_missing_collective_joins_precollective_failure_agreement(self):
        module = smoke()
        stages = []

        def gather(result, state):
            stages.append(state[0])
            result[:] = [state] * 4

        dist = SimpleNamespace(all_reduce=lambda *args, **kwargs: None,
                               all_gather_object=gather)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        runtime = SimpleNamespace(run=lambda *args: None)
        with patch.object(module, 'dist', dist, create=True), patch.object(module, 'torch', torch, create=True):
            with self.assertRaises(module.AgreedFailure):
                module.run_captured(runtime, (), None, 'case')
        self.assertEqual(stages, ['case:pre_collective', 'case:runtime_complete'])


if __name__ == '__main__':
    unittest.main()
