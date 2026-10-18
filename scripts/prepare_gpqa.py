#!/usr/bin/env python3
"""Prepare GPQA Main with the pinned DeepSeek-V4 Think High encoder.

The output contains decodable test token IDs and must stay in the ignored
data/gpqa/prepared directory. No question or answer text is printed.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.gpqa import PROMPT_PREFIX, build_message, ordered_indices
from src.weave.calibration import atomic_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model(model_dir: Path, manifest: dict) -> tuple:
    observed_hashes = {}
    for relative, expected in manifest["sha256"].items():
        path = model_dir / relative
        if not path.is_file():
            raise ValueError(f"Model metadata missing: {path}")
        actual = sha256(path)
        observed_hashes[relative] = actual
        if relative != "config.json" and actual != expected:
            raise ValueError(f"Model metadata SHA256 mismatch: {relative}: {actual}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["DeepseekV4ForCausalLM"] or config.get("expert_dtype") not in (None, "fp4"):
        raise ValueError("Expected original DeepSeek-V4-Flash with MXFP4 routed experts")
    return config, observed_hashes


def source_record(source_manifest: dict) -> dict:
    matches = [record for record in source_manifest["records"] if Path(record["path"]).name == "gpqa_main.csv"]
    if len(matches) != 1:
        raise ValueError("Source manifest must contain exactly one GPQA Main record")
    return matches[0]


def load_encoder(path: Path):
    spec = importlib.util.spec_from_file_location("weave_pinned_dsv4_encoding", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import DeepSeek encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def percentile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def write_jsonl_atomically(path: Path, records: list) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        # Preserve inherited GPUQ named-user ACL access across root containers.
        os.fchmod(stream.fileno(), 0o664)
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/gpqa/raw/gpqa_main.csv")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/gpqa/source_manifest.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/tokenizers/DeepSeek-V4-Flash")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "data/gpqa/deepseek_model_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/gpqa/prepared/deepseek-v4-flash")
    args = parser.parse_args()

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    config, observed_model_hashes = verify_model(args.model_dir, model_manifest)
    source = source_record(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    csv_hash = sha256(args.csv)
    if csv_hash != source["sha256"]:
        raise ValueError(f"GPQA Main CSV SHA256 mismatch: {csv_hash}")
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != source["rows"] or not rows:
        raise ValueError(f"GPQA Main row count mismatch: {len(rows)}")

    encode_messages = load_encoder(args.model_dir / "encoding/encoding_dsv4.py")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    records = []
    lengths = []
    for original_index in ordered_indices(len(rows)):
        message = build_message(rows[original_index], original_index)
        prompt = encode_messages(
            [{"role": "user", "content": message}], thinking_mode="thinking", reasoning_effort="high"
        )
        if not prompt.startswith("<｜begin▁of▁sentence｜>") or not prompt.endswith("<｜Assistant｜><think>"):
            raise ValueError("Unexpected DeepSeek Think High prompt format")
        input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not input_ids or input_ids[0] != config.get("bos_token_id"):
            raise ValueError("Tokenizer did not preserve the DeepSeek BOS token")
        lengths.append(len(input_ids))
        records.append({
            "row_index": original_index,
            "input_token_ids": input_ids,
            "input_tokens": len(input_ids),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "main.jsonl"
    write_jsonl_atomically(records_path, records)
    audit = {
        "count": len(lengths), "min": min(lengths), "median": statistics.median(lengths),
        "p95": percentile(lengths, 0.95), "max": max(lengths),
        "at_least_8192": sum(length >= 8192 for length in lengths),
    }
    manifest = {
        "model_id": model_manifest["model_id"],
        "model_revision": model_manifest["revision"],
        "pinned_model_metadata_sha256": model_manifest["sha256"],
        "observed_model_metadata_sha256": observed_model_hashes,
        "config_matches_pinned": observed_model_hashes["config.json"] == model_manifest["sha256"]["config.json"],
        "gpqa_main_sha256": csv_hash,
        "order_seed": 42,
        "choice_seed": "42 + original zero-based CSV row index",
        "prompt_instruction": PROMPT_PREFIX.strip(),
        "reasoning_mode": "Think High",
        "thinking_mode": "thinking",
        "reasoning_effort": "high",
        "max_new_tokens": 32768,
        "normal_eos_allowed": True,
        "record_sha256": sha256(records_path),
        **audit,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "audit": audit, "record_sha256": manifest["record_sha256"]}))


if __name__ == "__main__":
    main()
