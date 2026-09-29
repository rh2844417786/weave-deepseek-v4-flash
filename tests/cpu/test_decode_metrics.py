"""Decode throughput counts actual generated token IDs after the first token."""

import unittest

from src.weave.bench import decode_metrics


class DecodeMetricsTest(unittest.TestCase):
    def test_first_token_is_excluded_from_decode_rate(self):
        metrics = decode_metrics(output_tokens=5, first_token_s=10.0, last_token_s=12.0)
        self.assertEqual(metrics["decode_tokens"], 4)
        self.assertEqual(metrics["decode_seconds"], 2.0)
        self.assertEqual(metrics["decode_tps"], 2.0)
        self.assertEqual(metrics["tpot_ms"], 500.0)

    def test_single_token_has_no_decode_rate(self):
        metrics = decode_metrics(output_tokens=1, first_token_s=10.0, last_token_s=10.0)
        self.assertEqual(metrics["decode_tokens"], 0)
        self.assertIsNone(metrics["decode_tps"])
        self.assertIsNone(metrics["tpot_ms"])

    def test_rejects_nonpositive_decode_duration_for_multiple_tokens(self):
        with self.assertRaisesRegex(ValueError, "last_token_s"):
            decode_metrics(output_tokens=2, first_token_s=10.0, last_token_s=10.0)


if __name__ == "__main__":
    unittest.main()
