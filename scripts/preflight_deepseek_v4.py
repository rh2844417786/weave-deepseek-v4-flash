#!/usr/bin/env python3
"""Read-only inventory for a DeepSeek-V4-Flash checkpoint and SGLang host.

This does not load model weights or establish that SGLang can serve the model.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_checkpoint(model_path: Path, required_expert_dtype: str = "fp4") -> dict:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise ValueError(f"Required checkpoint metadata missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    architecture = (config.get("architectures") or [None])[0]
    if architecture != "DeepseekV4ForCausalLM" or config.get("model_type") != "deepseek_v4":
        raise ValueError(f"Unexpected model architecture: {architecture!r}")
    expert_dtype = config.get("expert_dtype", "unknown")
    if expert_dtype != required_expert_dtype:
        raise ValueError(
            f"Checkpoint expert_dtype is {expert_dtype!r}; required {required_expert_dtype!r}"
        )

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing or empty weight_map: {index_path}")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shard_names):
        raise ValueError("Checkpoint index contains an unsafe shard path")
    missing = [name for name in shard_names if not (model_path / name).is_file()]
    if missing:
        raise ValueError("Missing checkpoint shards: " + ", ".join(missing))

    return {
        "model_path": str(model_path),
        "architecture": architecture,
        "model_type": config["model_type"],
        "expert_dtype": expert_dtype,
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "quantization_config": config.get("quantization_config"),
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
        "shard_count": len(shard_names),
        "shard_bytes": sum((model_path / name).stat().st_size for name in shard_names),
        "missing_shards": [],
        "shard_check": "presence_and_size_only; shard contents not hashed",
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _gpu_inventory() -> dict:
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "error": str(exc)}
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
        devices.append(dict(zip(("index", "uuid", "name", "memory_mib"), fields)))
    return {"status": "observed", "devices": devices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-expert-dtype", default="fp4")
    args = parser.parse_args()

    report = {
        "checkpoint": inspect_checkpoint(args.model_path, args.required_expert_dtype),
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {name: _version(name) for name in ("sglang", "torch", "triton", "sgl-kernel")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_inventory": _gpu_inventory(),
        },
        "status": "inventory_only_not_generation_verified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote read-only inventory: {args.output}")


if __name__ == "__main__":
    main()
