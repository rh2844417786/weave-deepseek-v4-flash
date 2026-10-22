"""CPU checks cover layout and evidence policy; CUDA protocol needs GPU harness."""
import importlib
import unittest


class TransportContractTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('weave.native_tail_transport')
        except ModuleNotFoundError:
            self.fail('native tail transport contract is not implemented')

    def test_chunk_owner_layout_is_bijective_and_inverts_logical_output(self):
        shape = self.module().TransportShape(tokens=2, hidden=32, chunks=2)
        self.assertEqual(shape.channels_per_owner, 4)
        self.assertEqual(shape.payload_index(1, 2, 1, 3), 55)
        self.assertEqual(shape.output_index(1, 2, 1, 3), 59)
        payload = {shape.payload_index(c, o, t, x)
                   for c in range(2) for o in range(4)
                   for t in range(2) for x in range(4)}
        self.assertEqual(payload, set(range(64)))
        output = {shape.output_index(c, o, t, x)
                  for c in range(2) for o in range(4)
                  for t in range(2) for x in range(4)}
        self.assertEqual(output, set(range(64)))

    def test_empty_payload_still_has_four_owner_protocol_tasks(self):
        shape = self.module().TransportShape(tokens=0, hidden=32, chunks=1)
        self.assertEqual(shape.payload_elements, 0)
        self.assertEqual(shape.task_count, 5)

    def test_unsupported_layout_fails_before_cuda_allocation(self):
        module = self.module()
        for args in ((-1, 16, 1), (1, 15, 1), (1, 32, 3), (1, 0, 1), (1, 32, 1.0), (True, 32, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.TransportShape(*args)

    def test_finite_error_is_record_only_but_protocol_failure_is_fatal(self):
        assess = self.module().assess_transport_evidence
        evidence = {'abort_code': 0, 'sequence': 9, 'tasks': [1]*5,
                    'expected_tasks': 5, 'nonfinite': 0,
                    'acknowledged': True, 'canary_ok': True,
                    'relative_l2': 1000.0, 'max_abs': 100000.0}
        self.assertTrue(assess(evidence)['valid'])
        self.assertEqual(assess(evidence)['numerical_policy'], 'record_only')
        for key, value in [('tasks', [2, 0, 1, 1, 1]), ('abort_code', 1),
                           ('nonfinite', 1), ('acknowledged', False),
                           ('canary_ok', False)]:
            with self.subTest(key=key):
                self.assertFalse(assess(dict(evidence, **{key: value}))['valid'])


if __name__ == '__main__':
    unittest.main()
