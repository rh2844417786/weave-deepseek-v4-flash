"""The prepared GPQA manifest must be reproducible and contain no answer text."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "tokenizers package required")
class PrepareGPQACLITest(unittest.TestCase):
    def test_writes_fixed_order_token_ids_without_raw_answers(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            (model / "encoding").mkdir(parents=True)
            (model / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV4ForCausalLM"], "expert_dtype": "fp4", "bos_token_id": 0})
            )
            tokenizer = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(model / "tokenizer.json"))
            (model / "encoding/encoding_dsv4.py").write_text(
                "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
                "    assert thinking_mode == 'thinking' and reasoning_effort == 'high'\n"
                "    return '<｜begin▁of▁sentence｜><｜User｜>' + messages[0]['content'] + '<｜Assistant｜><think>'\n"
            )
            model_manifest = root / "model_manifest.json"
            model_manifest.write_text(json.dumps({
                "model_id": "test-deepseek",
                "revision": "fixed-test-revision",
                "sha256": {
                    name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                    for name in ("config.json", "tokenizer.json", "encoding/encoding_dsv4.py")
                },
            }))
            pinned_config_sha = json.loads(model_manifest.read_text())["sha256"]["config.json"]
            local_config = json.loads((model / "config.json").read_text())
            del local_config["expert_dtype"]
            (model / "config.json").write_text(json.dumps(local_config))

            csv_path = root / "gpqa_main.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "Question", "Correct Answer", "Incorrect Answer 1",
                    "Incorrect Answer 2", "Incorrect Answer 3", "Explanation",
                ])
                writer.writeheader()
                for i in range(2):
                    writer.writerow({
                        "Question": f"question {i}", "Correct Answer": f"correct {i}",
                        "Incorrect Answer 1": f"wrong a{i}",
                        "Incorrect Answer 2": f"wrong b{i}",
                        "Incorrect Answer 3": f"wrong c{i}",
                        "Explanation": f"private explanation {i}",
                    })
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(json.dumps({"records": [{
                "path": "data/gpqa/raw/gpqa_main.csv",
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "rows": 2,
            }]}))

            output = root / "prepared"
            command = [
                sys.executable, "scripts/prepare_gpqa.py", "--csv", str(csv_path),
                "--model-dir", str(model), "--model-manifest", str(model_manifest),
                "--source-manifest", str(source_manifest), "--output-dir", str(output),
            ]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in (output / "main.jsonl").read_text().splitlines()]
            self.assertEqual([record["row_index"] for record in records], [1, 0])
            self.assertTrue(all(record["input_token_ids"] for record in records))
            self.assertNotIn("private explanation", (output / "main.jsonl").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["count"], 2)
            self.assertEqual(manifest["reasoning_mode"], "Think High")
            self.assertEqual(manifest["max_new_tokens"], 32768)
            self.assertEqual(manifest["pinned_model_metadata_sha256"]["config.json"], pinned_config_sha)
            self.assertNotEqual(manifest["observed_model_metadata_sha256"]["config.json"], pinned_config_sha)


if __name__ == "__main__":
    unittest.main()
