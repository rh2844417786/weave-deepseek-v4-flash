#!/usr/bin/env python3
"""Offline, loopback-only C32 streaming measurement or deterministic acceptance."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.weave.c32_serving import (ACCEPTANCE_TOKENS, DATASET_SHA256, ClientError,
                                 generation_settings, load_dataset, run_acceptance,
                                 run_measurement, validate_base_url)
from src.weave.calibration import atomic_json


async def execute(args, workload):
    try:
        import aiohttp
    except ImportError:
        raise ClientError("aiohttp_missing_use_existing_SGLang_image_no_online_install") from None
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        # trust_env=False prevents proxy inheritance from routing localhost data
        # outside the machine. Redirects are disabled at every HTTP call.
        timeout = aiohttp.ClientTimeout(total=None, connect=30.0)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False,
                                        connector=aiohttp.TCPConnector(limit=64)) as session:
            if args.acceptance_only:
                return await run_acceptance(session, args.base_url, workload, args.output_dir,
                                            concurrency=args.concurrency)
            return await run_measurement(session, args.base_url, workload, args.output_dir,
                                         seconds=args.seconds, concurrency=args.concurrency,
                                         output_tokens=args.output_tokens)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=32, choices=[32])
    parser.add_argument("--output-tokens", type=int, default=1024, choices=[1024])
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    try:
        validate_base_url(args.base_url)
        workload = load_dataset(args.dataset)
        if not args.execute:
            print(json.dumps({"status": "preview_not_executed", **workload.identity,
                              "mode": "acceptance" if args.acceptance_only else "measurement",
                              "concurrency": args.concurrency,
                              "generation_settings": generation_settings(
                                  ACCEPTANCE_TOKENS if args.acceptance_only else args.output_tokens),
                              "requested_seconds": args.seconds}))
            return 0
        result = asyncio.run(execute(args, workload))
    except asyncio.CancelledError:
        print(json.dumps({"status": "failed", "error_codes": ["client_cancelled"]}))
        return 130
    except Exception as error:
        result = {"schema_version": 1, "status": "failed", "dataset_sha256": DATASET_SHA256,
                  "error_codes": [str(error) if isinstance(error, ClientError) else "client_setup_error"],
                  "raw_content_exported": False}
        if args.execute:
            name = "acceptance-summary.json" if args.acceptance_only else "client-summary.json"
            atomic_json(args.output_dir / name, result)
    print(json.dumps({"status": result["status"], "error_codes": result.get("error_codes", [])}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
