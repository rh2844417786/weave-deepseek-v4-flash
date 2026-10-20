#!/usr/bin/env python3
"""Preview or execute one bounded same-32-request native P0 diagnostic cohort."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.calibration import atomic_json
from src.weave.c32_serving import ClientError, validate_base_url
from src.weave.gpqa_c32 import load_gpqa_workload
from src.weave.native_p0_pair_client import (
    TIMEOUT_SECONDS, generation_settings, run_measurement, selection_identity)


async def execute(args, workload):
    try:
        import aiohttp
    except ImportError:
        raise ClientError('aiohttp_missing_use_existing_SGLang_image_no_online_install') from None
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30), trust_env=False,
                connector=aiohttp.TCPConnector(limit=64)) as session:
            return await run_measurement(
                session, args.base_url, workload, args.output_dir,
                timeout_seconds=args.timeout, controller_handshake=args.controller_handshake)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--timeout', type=float, default=TIMEOUT_SECONDS)
    parser.add_argument('--controller-handshake', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    try:
        if not args.records.is_absolute() or not args.output_dir.is_absolute():
            raise ClientError('absolute_records_and_output_dir_required')
        validate_base_url(args.base_url)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ClientError('invalid_timeout_seconds')
        workload = load_gpqa_workload(args.records)
        if not args.execute:
            print(json.dumps({'schema_version': 1, 'status': 'preview_not_executed',
                              'selection': selection_identity(workload),
                              'source_workload_identity': workload.identity,
                              'generation_settings': generation_settings(),
                              'request_count': 32, 'timeout_seconds': args.timeout,
                              'numerical_policy': 'record_only', 'raw_content_exported': False,
                              'required_server_flags': [
                                  '--incremental-streaming-output', '--stream-interval=1']}))
            return 0
        result = asyncio.run(execute(args, workload))
    except asyncio.CancelledError:
        print(json.dumps({'status': 'failed', 'error_codes': ['client_cancelled']}))
        return 130
    except Exception as error:
        code = str(error) if isinstance(error, ClientError) else (
            'prepared_GPQA_files_missing_use_offline_preparation'
            if isinstance(error, FileNotFoundError) else 'client_setup_error')
        result = {'schema_version': 1, 'status': 'failed', 'error_codes': [code],
                  'numerical_policy': 'record_only', 'raw_content_exported': False}
        if args.execute:
            atomic_json(args.output_dir / 'paired-client-summary.json', result)
    print(json.dumps({'status': result['status'], 'error_codes': result.get('error_codes', [])}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
