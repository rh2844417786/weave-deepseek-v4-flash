"""Checkpoint preflight must identify the stock mixed-precision checkpoint."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.preflight_deepseek_v4 import inspect_checkpoint


class CheckpointPreflightTest(unittest.TestCase):
    def make_checkpoint(self, root: Path, expert_dtype: str = "fp4") -> Path:
        model = root / "checkpoint"
        model.mkdir()
        (model / "model-00001-of-00002.safetensors").write_bytes(b"one")
        (model / "model-00002-of-00002.safetensors").write_bytes(b"two-two")
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DeepseekV4ForCausalLM"],
                    "model_type": "deepseek_v4",
                    "expert_dtype": expert_dtype,
                    "hidden_size": 4096,
                    "n_routed_experts": 256,
                    "num_experts_per_tok": 6,
                    "max_position_embeddings": 1048576,
                }
            )
        )
        (model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 10},
                    "weight_map": {
                        "model.layers.1.mlp.experts.0.gate_proj.weight": "model-00001-of-00002.safetensors",
                        "model.layers.1.mlp.experts.0.up_proj.weight": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )
        return model

    def test_reports_complete_fp4_checkpoint_without_reading_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_checkpoint(Path(tmp))
            report = inspect_checkpoint(model)
            self.assertEqual(report["architecture"], "DeepseekV4ForCausalLM")
            self.assertEqual(report["expert_dtype"], "fp4")
            self.assertEqual(report["shard_count"], 2)
            self.assertEqual(report["shard_bytes"], 10)
            self.assertEqual(report["missing_shards"], [])
            self.assertEqual(
                report["index_sha256"],
                hashlib.sha256((model / "model.safetensors.index.json").read_bytes()).hexdigest(),
            )

    def test_rejects_fp8_conversion_when_stock_fp4_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_checkpoint(Path(tmp), expert_dtype="fp8")
            with self.assertRaisesRegex(ValueError, "expert_dtype.*fp8.*fp4"):
                inspect_checkpoint(model)

    def test_rejects_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_checkpoint(Path(tmp))
            (model / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Missing.*model-00002"):
                inspect_checkpoint(model)


if __name__ == "__main__":
    unittest.main()
