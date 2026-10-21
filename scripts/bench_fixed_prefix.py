#!/usr/bin/env python3
"""Preview or execute an atomic fixed-prefix native SGLang B32 probe."""
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
from src.weave.fixed_prefix_data import (DEFAULT_STEPS, SUMMARY_FILE, TIMEOUT_SECONDS,
                                         load_teacher, run_phase, validate_pinned_selection)
from src.weave.gpqa_c32 import load_gpqa_workload


async def execute(args, workload):
    try:
        import aiohttp
    except ImportError:
        raise ClientError('aiohttp_missing_use_existing_SGLang_image_no_online_install') from None
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30), trust_env=False,
                connector=aiohttp.TCPConnector(limit=64)) as session:
            return await run_phase(session, args.base_url, workload, args.output_dir, args.teacher,
                                   phase=args.phase, steps=args.steps, timeout_seconds=args.timeout)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--phase', choices=('seed', 'measure', 'trace'), required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--timeout', type=float, default=TIMEOUT_SECONDS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    args.steps = args.steps if args.steps is not None else (2 if args.phase == 'trace' else DEFAULT_STEPS)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    try:
        if not all(path.is_absolute() for path in (args.records, args.output_dir, args.teacher)):
            raise ClientError('absolute_records_output_and_teacher_paths_required')
        if args.output_dir.resolve() in args.teacher.resolve().parents:
            raise ClientError('teacher_must_be_outside_public_client_directory')
        validate_base_url(args.base_url)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ClientError('invalid_timeout_seconds')
        if args.steps < 1:
            raise ClientError('positive_integer_steps_required')
        workload = load_gpqa_workload(args.records)
        selection = validate_pinned_selection(workload)
        if not args.execute:
            teacher_sha256 = None
            if args.phase != 'seed':
                teacher_sha256 = load_teacher(args.teacher,
                    selection_sha256=selection['selection_sha256'], steps=args.steps)['sha256']
            print(json.dumps({'schema_version': 1, 'status': 'preview_not_executed',
                              'phase': args.phase, 'steps': args.steps, 'selection': selection,
                              'source_workload_identity': workload.identity,
                              'input_tokens': 6059, 'max_input_tokens': 343,
                              'request_count': 32, 'atomic_generate_requests': 1,
                              'teacher_sha256': teacher_sha256, 'timeout_seconds': args.timeout,
                              'numerical_policy': 'record_only', 'raw_content_exported': False,
                              'cache_flush_owner': 'controller'}))
            return 0
        result = asyncio.run(execute(args, workload))
    except asyncio.CancelledError:
        print(json.dumps({'status': 'failed', 'error_codes': ['client_cancelled']}))
        return 130
    except Exception as error:
        code = str(error) if isinstance(error, ClientError) else (
            'prepared_GPQA_files_missing_use_offline_preparation'
            if isinstance(error, FileNotFoundError) else 'fixed_prefix_client_setup_error')
        result = {'schema_version': 1, 'status': 'failed', 'phase': args.phase,
                  'error_codes': [code], 'numerical_policy': 'record_only',
                  'raw_content_exported': False, 'cleanup_complete': True,
                  'abort_requests': 0, 'abort_failures': 0, 'requests_completed': 0}
        if args.execute:
            atomic_json(args.output_dir/SUMMARY_FILE, result)
    print(json.dumps({'status': result['status'], 'error_codes': result.get('error_codes', [])}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
