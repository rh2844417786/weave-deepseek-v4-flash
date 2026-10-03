import unittest
from src.weave.ops import ProfileBank


class ProfileBankTest(unittest.TestCase):
    def test_metadata_shape_selects_preallocated_bucket_without_clamping(self):
        bank=ProfileBank([(1,'decode'),(64,'small'),(2048,'prefill')])
        self.assertEqual(bank.for_tokens(0),'decode')
        self.assertEqual(bank.for_tokens(1),'decode')
        self.assertEqual(bank.for_tokens(2),'small')
        self.assertEqual(bank.for_tokens(64),'small')
        self.assertEqual(bank.for_tokens(65),'prefill')
        for n in (-1,2049):
            with self.assertRaises(ValueError): bank.for_tokens(n)
