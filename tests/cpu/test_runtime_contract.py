import tempfile
from pathlib import Path
import unittest

from src.weave.runtime_contract import load_contract, validate_inventory, verify_adapter_sources


class RuntimeContractTest(unittest.TestCase):
    def inventory(self):
        return {'python': '3.12.3', 'packages': {'torch': '2.11.0+cu130', 'sglang': '0.5.17',
                                              'transformers': '5.12.1'},
                'torch_cuda': '13.0', 'nvcc_release': '13.0', 'cuda_available': True,
                'devices': [{'capability': [9, 0]} for _ in range(4)]}

    def test_core_ignores_sglang_but_rejects_torch_drift(self):
        observed = self.inventory()
        observed['packages']['sglang'] = '0.5.18'
        observed['packages']['transformers'] = 'not_installed'
        self.assertEqual(validate_inventory(observed, 'core'), [])
        observed['packages']['torch'] = '2.13.0+cu130'
        self.assertTrue(any('torch' in error for error in validate_inventory(observed, 'core')))

    def test_adapter_requires_the_audited_sglang_and_transformers(self):
        self.assertEqual(validate_inventory(self.inventory(), 'adapter'), [])
        observed = self.inventory()
        observed['packages']['sglang'] = '0.5.16'
        self.assertTrue(any('sglang' in error for error in validate_inventory(observed, 'adapter')))

    def test_missing_compiler_or_gpu_is_a_recordable_failure(self):
        observed = self.inventory()
        observed['nvcc_release'] = 'unavailable'
        observed['devices'] = observed['devices'][:3]
        errors = validate_inventory(observed, 'core')
        self.assertTrue(any('nvcc' in error for error in errors))
        self.assertTrue(any('four' in error for error in errors))

    def test_source_hash_change_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'critical.py').write_text('patched')
            with self.assertRaisesRegex(ValueError, 'critical.py'):
                verify_adapter_sources(base, {'critical.py': 'wrong-sha'})

    def test_default_runtime_is_nearest_measured_image(self):
        contract = load_contract()
        self.assertEqual(contract['image'], 'lmsysorg/sglang:v0.5.17')
        self.assertEqual(contract['work_directory'], 'work')


if __name__ == '__main__':
    unittest.main()
