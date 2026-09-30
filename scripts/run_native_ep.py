#!/usr/bin/env python3
"""Preview or launch the native EP4 candidate under a GPUQ allocation.

This is a bring-up launcher, not the 1200-second performance experiment runner.
No readiness, generation, or EP-correctness claim follows from process exit.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

DEFAULT_IMAGE = 'sangfor-inference-sglang-nvidia-cuda13.0-ubuntu24.04-x86_64:v2.4.0.2'


def docker_command(args, image: str, name: str) -> list:
    return [
        'docker', 'run', '--rm', '--name', name,
        '--gpus', '"device=4,5,6,7"', '--network=host', '--shm-size=8g',
        '-v', f'{args.model_path}:/model:ro', '-v', f'{args.work_root}:/work',
        '-e', 'CUDA_VISIBLE_DEVICES=0,1,2,3', '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-e', 'NCCL_IB_DISABLE=1', '-e', 'NCCL_SOCKET_IFNAME=lo',
        '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'NCCL_CUMEM_HOST_ENABLE=0',
        '-e', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '-e', 'TORCH_NCCL_BLOCKING_WAIT=1',
        '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
        image, args.runtime_python, '-m', 'sglang.launch_server',
        '--model-path', '/model', '--trust-remote-code',
        '--served-model-name', 'weave-dsv4-native', '--host', '127.0.0.1', '--port', str(args.port),
        '--tp-size', '4', '--ep-size', '4',
        '--moe-runner-backend', 'marlin', '--moe-a2a-backend', 'none',
        '--context-length', '131072', '--mem-fraction-static', '0.80',
        '--chunked-prefill-size', '8192', '--max-running-requests', '1',
        '--cuda-graph-backend-decode', 'full', '--cuda-graph-max-bs-decode', '1',
        '--watchdog-timeout', '1800', '--disable-custom-all-reduce',
        '--disable-shared-experts-fusion', '--reasoning-parser', 'deepseek-v4',
        '--decode-log-interval', '10', '--enable-metrics',
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--runtime-python', default='/usr/bin/python3')
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--execute', action='store_true', help='launch Docker; default only prints the candidate command')
    args = parser.parse_args()
    allocation = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if allocation != '4,5,6,7':
        parser.error('GPUQ allocation must be CUDA_VISIBLE_DEVICES=4,5,6,7')
    if not args.model_path.is_absolute() or not args.work_root.is_absolute():
        parser.error('model-path and work-root must be absolute paths')
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    name = 'weave-native-ep-' + uuid.uuid4().hex[:12]
    if not args.execute:
        print(json.dumps({'status': 'preview_not_executed', 'docker_argv': docker_command(args, args.image, name)}, indent=2))
        return 0
    if not args.model_path.is_dir():
        parser.error(f'model directory missing: {args.model_path}')
    args.work_root.mkdir(parents=True, exist_ok=True)
    inspection = subprocess.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                                capture_output=True, text=True, check=True, timeout=30)
    image_id = inspection.stdout.strip()
    if not image_id.startswith('sha256:'):
        raise ValueError('docker image inspect did not return a content-addressed image ID')
    run_dir = args.work_root / ('native-ep-' + uuid.uuid4().hex)
    run_dir.mkdir()
    command = docker_command(args, image_id, name)
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'requested_image': args.image, 'resolved_image_id': image_id,
        'host_cuda_visible_devices': allocation,
        'gpuq_job_id': os.environ.get('GPUQ_JOB_ID'),
        'docker_argv': command, 'generation_status': 'not_verified',
        'ep_correctness_status': 'not_verified', 'performance_status': 'not_run',
        'exit_code': None,
    }
    report_path = run_dir / 'launch.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Native EP4 candidate; log: {run_dir / "server.log"}', flush=True)
    with (run_dir / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=30)
            process.wait(timeout=30)
            code = 130
    report['exit_code'] = code
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
