import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from src.weave.ops import extension_name, kernel_source_sha256


class ExtensionIdentityTest(unittest.TestCase):
    def test_expert_rounding_header_change_invalidates_build_and_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'csrc').mkdir()
            for name in ('megakernel.cu', 'scheduler.h', 'expert_numerics.h'):
                (root/'csrc'/name).write_text(name)
            with patch('src.weave.ops.__file__', str(root/'src/weave/ops.py')):
                before = (kernel_source_sha256(), extension_name('torch', 'cuda'))
                (root/'csrc/expert_numerics.h').write_text('changed rounding')
                after = (kernel_source_sha256(), extension_name('torch', 'cuda'))
                self.assertNotEqual(before[0], after[0])
                self.assertNotEqual(before[1], after[1])

    def test_torch_or_cuda_change_cannot_reuse_an_extension_name(self):
        original = extension_name('2.11.0+cu130', '13.0')
        self.assertNotEqual(original, extension_name('2.13.0+cu130', '13.0'))
        self.assertNotEqual(original, extension_name('2.11.0+cu130', '12.8'))
        self.assertEqual(original, extension_name('2.11.0+cu130', '13.0'))


if __name__ == '__main__':
    unittest.main()
