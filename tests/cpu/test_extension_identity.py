import unittest
from src.weave.ops import extension_name


class ExtensionIdentityTest(unittest.TestCase):
    def test_torch_or_cuda_change_cannot_reuse_an_extension_name(self):
        original = extension_name('2.11.0+cu130', '13.0')
        self.assertNotEqual(original, extension_name('2.13.0+cu130', '13.0'))
        self.assertNotEqual(original, extension_name('2.11.0+cu130', '12.8'))
        self.assertEqual(original, extension_name('2.11.0+cu130', '13.0'))


if __name__ == '__main__':
    unittest.main()
