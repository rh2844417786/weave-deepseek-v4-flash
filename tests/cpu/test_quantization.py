import math
import unittest
from src.weave.quantization import decode_row


class MXFP4Test(unittest.TestCase):
    def test_low_nibble_first_signed_zero_and_block_scale(self):
        values = decode_row(bytes([0x21, 0x98] + [0x00] * 14 + [0x76] * 16), bytes([127, 128]))
        self.assertEqual(values[:4], (.5, 1., -0., -.5))
        self.assertEqual(math.copysign(1., values[2]), -1.)
        self.assertEqual(values[32:34], (8., 12.))

    def test_missing_or_nan_scale_is_rejected(self):
        for scales in (b'', bytes([255])):
            with self.assertRaises(ValueError):
                decode_row(bytes(16), scales)


if __name__ == '__main__':
    unittest.main()
