#!/usr/bin/env python3
"""Record the actual container environment before compilation or model launch."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.runtime_contract import load_contract, validate_inventory, verify_adapter_sources
from src.weave.ops import kernel_source_sha256


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = {'sgl-kernel': 'sgl_kernel', 'flashinfer-python': 'flashinfer'}.get(name, name.replace('-', '_'))
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        return f'no_distribution_metadata; module found at {spec.origin}' if spec else 'not_installed'


def capture():
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'kernel_source_sha256': kernel_source_sha256(),
              'container_hostname': platform.node(), 'host_hostname': os.environ.get('WEAVE_HOST_HOSTNAME'),
              'requested_image': os.environ.get('WEAVE_IMAGE_REF'),
              'resolved_image_id': os.environ.get('WEAVE_IMAGE_ID'),
              'host_gpu_allocation': os.environ.get('WEAVE_HOST_ALLOCATION'),
              'python': platform.python_version(), 'python_executable': sys.executable,
              'packages': {name: package_version(name) for name in
                           ('torch', 'sglang', 'transformers', 'triton', 'sgl-kernel', 'flashinfer-python')},
              'torch_cuda': None, 'cuda_available': False, 'devices': [], 'nvcc_release': 'unavailable'}
    cuda_home = os.environ.get('CUDA_HOME')
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        result['packages']['torch'] = str(torch.__version__)
        result['torch_cuda'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        cuda_home = CUDA_HOME
        if result['cuda_available']:
            for device in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(device)
                result['devices'].append({'ordinal': device, 'name': prop.name,
                    'capability': [prop.major, prop.minor], 'memory_bytes': prop.total_memory})
    except Exception as exc:
        result['torch_probe_error'] = f'{type(exc).__name__}: {exc}'
    nvcc = str(Path(cuda_home)/'bin/nvcc') if cuda_home else shutil.which('nvcc')
    result['cuda_home'], result['nvcc_executable'] = cuda_home, nvcc
    if nvcc:
        try:
            output = subprocess.run([nvcc, '--version'], capture_output=True, text=True, check=True, timeout=20).stdout
            result['nvcc_output'] = output.strip()
            match = re.search(r'release\s+(\d+\.\d+)', output)
            result['nvcc_release'] = match.group(1) if match else 'unrecognized'
        except (OSError, subprocess.SubprocessError) as exc:
            result['nvcc_probe_error'] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-image', action='store_true')
    parser.add_argument('--stage', choices=['core', 'native', 'adapter'], default='core')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run-core-tests', action='store_true')
    parser.add_argument('--run-checkpoint-tests', action='store_true')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--checkpoint-layers', default='all')
    parser.add_argument('--exec-command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    contract = load_contract()
    if args.print_image:
        print(contract['image'])
        return 0
    if args.output_dir is None:
        parser.error('--output-dir required')
    if args.run_core_tests and (args.stage != 'core' or args.exec_command):
        parser.error('--run-core-tests is only valid for the core stage without another command')
    if args.run_checkpoint_tests and (args.stage != 'core' or args.exec_command or args.model_path is None):
        parser.error('--run-checkpoint-tests requires core stage, model-path and no exec-command')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed = capture()
    errors = validate_inventory(observed, args.stage, contract)
    if args.stage in ('native', 'adapter') and not errors:
        try:
            observed['adapter_source_sha256'] = verify_adapter_sources()
        except ValueError as exc:
            errors.append(str(exc))
    observed.update({'validation_stage': args.stage, 'validation_errors': errors,
                     'runtime_status': 'failed' if errors else 'passed', 'expected_contract': contract})
    path = args.output_dir/'runtime.json'
    path.write_text(json.dumps(observed, indent=2)+'\n')
    print(json.dumps({'runtime_status': observed['runtime_status'], 'errors': errors, 'evidence': str(path)}), flush=True)
    if errors:
        return 2
    if args.run_core_tests or args.run_checkpoint_tests:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                   str(ROOT/'tests/gpu/check_functional.py'), '--output-dir', str(args.output_dir)]
    else:
        command = args.exec_command
    if not command:
        return 0
    code = subprocess.call(command, cwd=ROOT)
    observed['child_exit_code'] = code
    if args.run_core_tests or args.run_checkpoint_tests:
        ranks = []
        for rank in range(4):
            report = args.output_dir/f'rank{rank}.json'
            try:
                ranks.append(json.loads(report.read_text()).get('status') == 'functional_checks_passed')
            except (OSError, ValueError):
                ranks.append(False)
        observed['core_validation_status'] = 'passed' if code == 0 and all(ranks) else 'failed'
        if code == 0 and not all(ranks):
            code = 3
    if code == 0 and args.run_checkpoint_tests:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                   str(ROOT/'tests/gpu/check_checkpoint.py'), '--output-dir', str(args.output_dir),
                   '--model-path', str(args.model_path), '--layers', args.checkpoint_layers]
        code = subprocess.call(command, cwd=ROOT)
        ranks = []
        for rank in range(4):
            try:
                data = json.loads((args.output_dir/f'checkpoint-rank{rank}.json').read_text())
                expected_layers = list(range(43)) if args.checkpoint_layers == 'all' else [int(v) for v in args.checkpoint_layers.split(',')]
                ranks.append(data.get('status') == 'checkpoint_checks_passed' and data['layers'] == expected_layers
                             and data['kernels'] == ['scalar', 'wmma', 'tiled'])
            except (OSError, ValueError, KeyError):
                ranks.append(False)
        observed['checkpoint_validation_status'] = 'passed' if code == 0 and all(ranks) else 'failed'
        if code == 0 and not all(ranks):
            code = 3
        observed['child_exit_code'] = code
    path.write_text(json.dumps(observed, indent=2)+'\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
