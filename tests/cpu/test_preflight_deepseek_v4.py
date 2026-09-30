"""Checkpoint preflight must identify the stock mixed-precision checkpoint."""

import hashlib
import json
from pathlib import Path
import struct
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

    def test_derives_mxfp4_experts_when_config_omits_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_checkpoint(Path(tmp))
            config_path = model / "config.json"
            config = json.loads(config_path.read_text())
            del config["expert_dtype"]
            config_path.write_text(json.dumps(config))
            expert_name = "model.layers.3.ffn.experts.0.w1.weight"
            scale_name = "model.layers.3.ffn.experts.0.w1.scale"
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {expert_name: "model-00001-of-00002.safetensors", scale_name: "model-00001-of-00002.safetensors"}
            }))
            header = json.dumps({
                expert_name: {"dtype": "I8", "shape": [64, 1024], "data_offsets": [0, 1]},
                scale_name: {"dtype": "F8_E8M0", "shape": [2, 64], "data_offsets": [1, 2]},
            }).encode()
            (model / "model-00001-of-00002.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"xx")
            report = inspect_checkpoint(model)
            self.assertEqual(report["expert_dtype"], "fp4")
            self.assertEqual(report["expert_dtype_source"], "derived_from_shard_header")
            self.assertEqual(report["expert_dtype_evidence"]["stored_width"], 1024)


if __name__ == "__main__":
    unittest.main()
