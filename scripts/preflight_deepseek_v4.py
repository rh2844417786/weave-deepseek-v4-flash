#!/usr/bin/env python3
"""Read-only inventory for a DeepSeek-V4-Flash checkpoint and SGLang host.

This does not load model weights or establish that SGLang can serve the model.
At most one shard header is read, to classify the routed-expert precision when
the checkpoint config omits ``expert_dtype``.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_safetensors_header(path: Path) -> dict:
    """Read a safetensors header only; the weight payload is never read."""
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    header.pop("__metadata__", None)
    return header


def derive_expert_dtype(model_path: Path, shard_name: str, block_size: int = 32) -> dict:
    """Classify routed-expert precision from one shard header alone.

    Published DeepSeek-V4-Flash configs carry ``expert_dtype``, but some server
    copies omit the field. The routed experts are then classified by comparing
    the stored width of an expert matrix with its block-scale width: a stored
    byte holding two FP4 values with block size 32 is MXFP4, an unpacked matrix
    is FP8.
    """
    header = _read_safetensors_header(model_path / shard_name)
    for name in sorted(header):
        if ".ffn.experts." not in name or not name.endswith(".weight"):
            continue
        scale = header.get(name[: -len("weight")] + "scale")
        if scale is None:
            continue
        stored_width = header[name]["shape"][-1]
        stored_dtype = header[name]["dtype"]
        packed = stored_width * 2 == scale["shape"][-1] * block_size
        unpacked = stored_width == scale["shape"][-1] * block_size
        if stored_dtype in ("I8", "U8") and packed:
            derived = "fp4"
        elif stored_dtype in ("F8_E4M3", "F8_E5M2") and unpacked:
            derived = "fp8"
        else:
            derived = "unknown"
        return {
            "expert_tensor": name,
            "storage_dtype": stored_dtype,
            "scale_dtype": scale["dtype"],
            "stored_width": stored_width,
            "scale_width": scale["shape"][-1],
            "assumed_block_size": block_size,
            "derived_expert_dtype": derived,
        }
    raise ValueError(f"No routed-expert weight found in shard header: {shard_name}")


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
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing or empty weight_map: {index_path}")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shard_names):
        raise ValueError("Checkpoint index contains an unsafe shard path")
    missing = [name for name in shard_names if not (model_path / name).is_file()]
    if missing:
        raise ValueError("Missing checkpoint shards: " + ", ".join(missing))

    expert_dtype = config.get("expert_dtype")
    expert_dtype_source = "config"
    expert_dtype_evidence = None
    if expert_dtype is None:
        expert_shard = next(
            (name for key, name in weight_map.items() if ".ffn.experts." in key), None
        )
        if expert_shard is None:
            raise ValueError("Checkpoint index lists no routed-expert weight")
        expert_dtype_evidence = derive_expert_dtype(model_path, expert_shard)
        expert_dtype = expert_dtype_evidence["derived_expert_dtype"]
        expert_dtype_source = "derived_from_shard_header"
    if expert_dtype != required_expert_dtype:
        raise ValueError(
            f"Checkpoint expert_dtype is {expert_dtype!r} "
            f"(source: {expert_dtype_source}); required {required_expert_dtype!r}"
        )

    return {
        "model_path": str(model_path),
        "architecture": architecture,
        "model_type": config["model_type"],
        "expert_dtype": expert_dtype,
        "expert_dtype_source": expert_dtype_source,
        "expert_dtype_evidence": expert_dtype_evidence,
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
        pass
    # Some images ship a package as a bare module with no distribution
    # metadata. Report where it was found rather than claiming it is absent.
    try:
        spec = importlib.util.find_spec(distribution.replace("-", "_"))
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return "not_installed"
    return f"no_distribution_metadata; module found at {spec.origin}"


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
    parser.add_argument(
        "--runtime-image",
        default=None,
        help="container image reference that provides the runtime, recorded verbatim",
    )
    parser.add_argument(
        "--runtime-provenance-note",
        default=None,
        help="provenance caveat about the runtime above, recorded verbatim",
    )
    parser.add_argument("--runtime-image-id", default=None)
    args = parser.parse_args()

    report = {
        "checkpoint": inspect_checkpoint(args.model_path, args.required_expert_dtype),
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "image": args.runtime_image,
            "image_id": args.runtime_image_id,
            "host_hostname": os.environ.get("WEAVE_HOST_HOSTNAME"),
            "provenance_note": args.runtime_provenance_note,
            "packages": {name: _version(name) for name in ("sglang", "torch", "triton", "sgl-kernel")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "gpu_inventory": _gpu_inventory(),
        },
        "status": "inventory_only_not_generation_verified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote read-only inventory: {args.output}")


if __name__ == "__main__":
    main()
