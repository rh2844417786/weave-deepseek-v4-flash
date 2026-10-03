#!/usr/bin/env python3
"""Preview or launch native/functional-Weave EP4 under a GPUQ allocation.

This is a bring-up launcher, not the 1200-second performance experiment runner.
No readiness, generation, or EP-correctness claim follows from process exit.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.runtime_contract import load_contract
from src.weave.container_lifecycle import run_container
CONTRACT = load_contract()
DEFAULT_IMAGE = CONTRACT['image']


def docker_command(args, image: str, name: str, run_name='preview') -> list:
    docker = [
        'docker', 'run', '--rm', '--pull=never', '--name', name,
        '--gpus', '"device=4,5,6,7"', '--network=host', '--shm-size=8g',
        '-v', f'{args.model_path}:/model:ro', '-v', f'{args.work_root}:/work',
        '-v', f'{ROOT}:/workspace:ro', '-w', '/workspace',
        '-e', 'CUDA_VISIBLE_DEVICES=0,1,2,3', '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-e', 'NCCL_IB_DISABLE=1', '-e', 'NCCL_SOCKET_IFNAME=lo',
        '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'NCCL_CUMEM_HOST_ENABLE=0',
        '-e', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '-e', 'TORCH_NCCL_BLOCKING_WAIT=1',
        '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
        '-e', 'HF_DATASETS_OFFLINE=1',
        '-e', f'WEAVE_IMAGE_REF={args.image}', '-e', f'WEAVE_IMAGE_ID={image}',
        '-e', f'WEAVE_HOST_HOSTNAME={os.uname().nodename}', '-e', 'WEAVE_HOST_ALLOCATION=4,5,6,7',
    ]
    functional = args.backend == 'weave_functional'
    if args.checkpoint_content_sha256:
        docker += ['-e', f'WEAVE_CHECKPOINT_CONTENT_SHA256={args.checkpoint_content_sha256}']
    if args.request_timing:
        docker += ['-e','WEAVE_REQUEST_TIMING=1','-e','PYTHONPATH=/workspace/timing_hooks:/workspace/src']
    elif functional:
        docker += ['-e','PYTHONPATH=/workspace/src']
    if functional:
        docker += [
            '-e', 'SGLANG_EXTERNAL_MODEL_PACKAGE=weave.sglang_ext.models',
            '-e', 'WEAVE_MODEL_PATH=/model', '-e', 'WEAVE_CAPACITY=2048',
            '-e', f'WEAVE_FIXED_PLAN={args.fixed_plan}', '-e', 'TORCH_EXTENSIONS_DIR=/work/torch_extensions',
            '-e', f'WEAVE_KERNEL={args.kernel}', '-e', f'WEAVE_DECODE_GRAPH={int(args.decode_graph == "full")}',
            '-e', f'WEAVE_EVIDENCE_PORT_BASE={args.evidence_port_base}',
        ]
        if args.profile:
            relative = args.profile.relative_to(args.work_root)
            docker += ['-e', f'WEAVE_PROFILE=/work/{relative}']
    command = docker + [image, args.runtime_python, '/workspace/scripts/check_runtime.py',
        '--stage', 'adapter' if functional else 'native', '--output-dir', f'/work/{run_name}',
        '--exec-command', args.runtime_python, '-m', 'sglang.launch_server',
        '--model-path', '/model', '--trust-remote-code',
        '--served-model-name', 'weave-dsv4-functional' if functional else 'weave-dsv4-native',
        '--host', '127.0.0.1', '--port', str(args.port),
        '--tp-size', '4', '--ep-size', '4',
        '--moe-runner-backend', 'marlin', '--moe-a2a-backend', 'none',
        '--context-length', '131072', '--mem-fraction-static', '0.80',
        '--chunked-prefill-size', '8192', '--max-running-requests', '1',
        '--cuda-graph-backend-decode', args.decode_graph, '--cuda-graph-max-bs-decode', '1',
        '--watchdog-timeout', '1800', '--disable-custom-all-reduce',
        '--disable-shared-experts-fusion', '--reasoning-parser', 'deepseek-v4',
        '--decode-log-interval', '10', '--enable-metrics',
    ]
    if args.request_timing:
        command += ['--stream-interval','1','--incremental-streaming-output']
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/CONTRACT['work_directory'])))
    parser.add_argument('--image', default=os.environ.get('IMAGE', DEFAULT_IMAGE))
    parser.add_argument('--runtime-python', default=os.environ.get('RUNTIME_PYTHON', CONTRACT['python_executable']))
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--backend', choices=['native', 'weave_functional'], default='native')
    parser.add_argument('--kernel', choices=['scalar', 'wmma', 'tiled'], default='scalar')
    parser.add_argument('--decode-graph', choices=['disabled', 'full'], help='functional graph requires explicit full opt-in')
    parser.add_argument('--fixed-plan', default='1,1,0', help='explicit c,K,completed-task steal budget per comm worker')
    parser.add_argument('--evidence-port-base', type=int, default=30110)
    parser.add_argument('--run-name', help='fresh native-ep-* directory name, for acceptance orchestration')
    parser.add_argument('--profile', type=Path, help='measured profile JSON under WORK_ROOT; otherwise functional fixed plan')
    parser.add_argument('--request-timing', action='store_true', help='shared native token-ready timing for formal groups')
    parser.add_argument('--checkpoint-content-sha256', help='verified original checkpoint digest for measured tiled profiles')
    parser.add_argument('--execute', action='store_true', help='launch Docker; default only prints the candidate command')
    args = parser.parse_args()
    if args.checkpoint_content_sha256 and not re.fullmatch(r'[0-9a-f]{64}', args.checkpoint_content_sha256):
        parser.error('checkpoint-content-sha256 must be a full SHA256 digest')
    args.decode_graph = args.decode_graph or ('disabled' if args.backend == 'weave_functional' else 'full')
    if args.run_name and not re.fullmatch(r'native-ep-[A-Za-z0-9_.-]+', args.run_name):
        parser.error('run-name must be a safe native-ep-* basename')
    try:
        c, k, steal = [int(v) for v in args.fixed_plan.split(',')]
        if c < 0 or not 1 <= k <= 8 or steal < 0:
            raise ValueError
    except ValueError:
        parser.error('fixed-plan must contain nonnegative c/steal and K in 1..8')
    if not 1 <= args.evidence_port_base <= 65532 or args.port in range(args.evidence_port_base, args.evidence_port_base+4):
        parser.error('evidence port range must be valid and distinct from serving port')
    allocation = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if allocation != '4,5,6,7':
        parser.error('GPUQ allocation must be CUDA_VISIBLE_DEVICES=4,5,6,7')
    if not args.model_path.is_absolute() or not args.work_root.is_absolute():
        parser.error('model-path and work-root must be absolute paths')
    args.work_root = args.work_root.resolve()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.profile:
        if args.backend != 'weave_functional' or not args.profile.is_absolute() or '..' in args.profile.parts:
            parser.error('profile requires weave_functional and an absolute path under WORK_ROOT')
        args.profile = args.profile.resolve()
        try:
            args.profile.relative_to(args.work_root)
        except ValueError:
            parser.error('profile must be under WORK_ROOT')
    if args.profile and args.kernel == 'tiled' and not args.checkpoint_content_sha256:
        parser.error('measured tiled profile requires checkpoint-content-sha256 from the passed core gate')
    name = 'weave-native-ep-' + uuid.uuid4().hex[:12]
    if not args.execute:
        print(json.dumps({'status': 'preview_not_executed', 'docker_argv': docker_command(args, args.image, name)}, indent=2))
        return 0
    if not args.model_path.is_dir():
        parser.error(f'model directory missing: {args.model_path}')
    args.work_root.mkdir(parents=True, exist_ok=True)
    if not os.access(args.work_root, os.W_OK | os.X_OK):
        parser.error('WORK_ROOT must be writable and searchable')
    inspection = subprocess.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                                capture_output=True, text=True, check=True, timeout=30)
    image_id = inspection.stdout.strip()
    if not image_id.startswith('sha256:'):
        raise ValueError('docker image inspect did not return a content-addressed image ID')
    run_dir = args.work_root / (args.run_name or 'native-ep-' + uuid.uuid4().hex)
    run_dir.mkdir()
    command = docker_command(args, image_id, name, run_dir.name)
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'requested_image': args.image, 'resolved_image_id': image_id,
        'host_cuda_visible_devices': allocation,
        'gpuq_job_id': os.environ.get('GPUQ_JOB_ID'),
        'docker_argv': command, 'generation_status': 'not_verified',
        'backend': args.backend,
        'runtime_contract': CONTRACT, 'runtime_report': str(run_dir/'runtime.json'),
        'ep_correctness_status': 'not_verified', 'performance_status': 'not_run',
        'exit_code': None,
    }
    report_path = run_dir / 'launch.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{args.backend} EP4 candidate; log: {run_dir / "server.log"}', flush=True)
    with (run_dir / 'server.log').open('w') as log:
        lifecycle = run_container(command, name, run_dir/'container_lifecycle.json', stdout=log,
                                  exit_code_path=run_dir/'exit_code.txt')
    code = lifecycle['exit_code']
    report['exit_code'] = code
    report['termination_signal'] = lifecycle['termination_signal']
    report['container_cleanup'] = lifecycle['cleanup_status']
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
