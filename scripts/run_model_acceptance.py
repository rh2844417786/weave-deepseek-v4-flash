#!/usr/bin/env python3
"""GPUQ job: checkpoint gate, native EP4, Weave eager and batch-one graph gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.ops import kernel_source_sha256
from src.weave.service_acceptance import validate_checkpoint_gate
from src.weave.checkpoint_identity import checkpoint_identity, checkpoint_stats
from scripts.check_sglang_service import wait_ready, check_service
from src.weave.gpu_allocation import allocated_gpus


def assert_ports_free(ports):
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))


class OwnedChild:
    """Register a private process group before allowing cancellation to unwind."""
    def __init__(self, cleanup_timeout=8):
        self.cleanup_timeout=cleanup_timeout
        self.process = self.lifecycle = None
        self.protected = False
        self.requested = None

    def signal(self, signum, frame):
        if self.requested is not None:
            return
        self.requested = signum
        if not self.protected:
            raise InterruptedError(f'GPUQ signal {signum}; cleaning current launcher')

    def check_cancelled(self):
        if self.requested is not None:
            raise InterruptedError(f'GPUQ signal {self.requested}; cleaning current launcher')

    def spawn(self, command, lifecycle, **kwargs):
        if self.process is not None:
            raise RuntimeError('previous child must be cleaned before starting the next')
        self.protected = True
        try:
            self.lifecycle = lifecycle
            self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
        finally:
            self.protected = False
        self.check_cancelled()  # Child is already registered if spawn was interrupted.
        return self.process

    def stop(self):
        process = self.process
        if process is None:
            return
        self.protected = True
        try:
            deadline = time.monotonic()+self.cleanup_timeout
            # The core bash and manager share this owned group. The Docker
            # client itself has a different session; its manager stops Docker.
            if process.poll() is None or not self.lifecycle.exists():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            process.wait(timeout=max(.1, deadline-time.monotonic()))
            # A bash leader can exit before its manager. Do not return until
            # that manager has recorded cleanup of its daemon-owned container.
            while True:
                try:
                    evidence = json.loads(self.lifecycle.read_text())
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    # A descendant may have created/truncated the report but
                    # not finished publishing it after the leader has exited.
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'complete child cleanup evidence missing: {self.lifecycle}')
                    time.sleep(.05)
            state = evidence.get('cleanup_status')
            if state not in ('absent', 'stopped', 'not_required'):
                raise RuntimeError(f'child cleanup unconfirmed: {state}; {self.lifecycle}')
            self.process = self.lifecycle = None
        finally:
            self.protected = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/'work')))
    parser.add_argument('--records', type=Path, default=ROOT/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl')
    parser.add_argument('--core-run-dir', type=Path, help='reuse a passed current-source all-43 checkpoint run')
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--evidence-port-base', type=int, default=30110)
    parser.add_argument('--kernel', choices=['scalar', 'wmma', 'tiled'], default='wmma')
    parser.add_argument('--controller-lifecycle', type=Path, help='outer orchestrator cleanup handshake file')
    parser.add_argument('--profile', type=Path, help='measured profile bank for dynamic tiled model validation')
    args = parser.parse_args()
    try: allocation=allocated_gpus()
    except ValueError as error: parser.error(str(error))
    os.environ['CUDA_VISIBLE_DEVICES']=allocation.csv
    if not args.work_root.is_absolute() or not args.model_path.is_absolute():
        parser.error('model-path and work-root must be absolute')
    if not args.records.is_file():
        parser.error('prepare the pinned Think High GPQA token-ID manifest first')
    if not 1 <= args.port <= 65535 or not 1 <= args.evidence_port_base <= 65532:
        parser.error('serving/evidence ports are outside the valid range')
    if args.port in range(args.evidence_port_base, args.evidence_port_base+4):
        parser.error('serving/evidence port ranges must be distinct')
    args.work_root = args.work_root.resolve()
    if args.controller_lifecycle:
        if not args.controller_lifecycle.is_absolute(): parser.error('controller lifecycle must be absolute')
        args.controller_lifecycle=args.controller_lifecycle.resolve()
        try: args.controller_lifecycle.relative_to(args.work_root)
        except ValueError: parser.error('controller lifecycle must be under work-root')
        if args.controller_lifecycle.exists(): parser.error('fresh controller lifecycle path required')
    if args.profile:
        args.profile = args.profile.resolve()
        if args.kernel != 'tiled' or not args.profile.is_dir():
            parser.error('profile requires tiled and an existing profile-bank directory')
        try: args.profile.relative_to(args.work_root)
        except ValueError: parser.error('profile must be under work-root')
    args.work_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    out = args.work_root/f'model-acceptance-{run_id}'
    out.mkdir()
    report = {'status': 'running', 'source_sha256': kernel_source_sha256(),
              'physical_gpus':list(allocation.ids),'host_gpu_allocation':allocation.csv,
              'kernel': args.kernel, 'scope': 'correctness_and_graph_acceptance', 'performance': 'not_run',
              'steps': {}, 'server_runs': {},
              'acceptance_policy': {'logprob': 'record_only', 'top_logprob_support': 'record_only'}}
    report_path = out/'acceptance.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Model acceptance evidence: {out}', flush=True)
    child = OwnedChild()
    previous = {sig: signal.signal(sig, child.signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        core = args.core_run_dir or args.work_root/f'cuda-functional.{run_id}'
        if args.core_run_dir is None:
            env = {**os.environ, 'CHECKPOINT_TESTS': '1', 'CHECKPOINT_LAYERS': 'all',
                   'MODEL_PATH': str(args.model_path), 'WORK_ROOT': str(args.work_root), 'CORE_RUN_NAME': core.name}
            with (out/'core-launcher.log').open('w') as log:
                process = child.spawn(['bash', str(ROOT/'scripts/run_cuda_functional_gpuq.sh')],
                                      core/'container_lifecycle.json', env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                    if code != 0:
                        raise subprocess.CalledProcessError(code, process.args)
                finally:
                    child.stop()
                child.check_cancelled()
        runtime = json.loads((core/'runtime.json').read_text())
        ranks = [json.loads((core/f'checkpoint-rank{rank}.json').read_text()) for rank in range(4)]
        index_sha = hashlib.sha256((args.model_path/'model.safetensors.index.json').read_bytes()).hexdigest()
        identity = checkpoint_identity(args.model_path, progress=lambda n, total, name:
            print(json.dumps({'checkpoint_hash_progress': [n, total], 'file': name}), flush=True))
        initial_stats = identity['verified_stats']
        (out/'checkpoint_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
        report['steps']['checkpoint'] = validate_checkpoint_gate(runtime, ranks,
            source_sha256=report['source_sha256'], index_sha256=index_sha, checkpoint_sha256=identity['sha256'])
        report['core_run_dir'] = str(core)
        native = out/'native-request.json'
        for label, backend, graph in [('native', 'native', 'full'),
                                      ('weave_eager', 'weave_functional', 'disabled'),
                                      ('weave_graph', 'weave_functional', 'full')]:
            assert_ports_free([args.port]+list(range(args.evidence_port_base, args.evidence_port_base+4)))
            name = f'native-ep-{run_id}-{label}'
            command = [sys.executable, str(ROOT/'scripts/run_native_ep.py'), '--execute',
                       '--model-path', str(args.model_path), '--work-root', str(args.work_root),
                       '--backend', backend, '--kernel', args.kernel, '--decode-graph', graph,
                       '--port', str(args.port), '--evidence-port-base', str(args.evidence_port_base), '--run-name', name]
            if args.profile and backend == 'weave_functional':
                command += ['--profile', str(args.profile), '--checkpoint-content-sha256', identity['sha256']]
            report['server_runs'][label] = str(args.work_root/name)
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            with (out/f'{label}-launcher.log').open('w') as log:
                process = child.spawn(command, args.work_root/name/'container_lifecycle.json',
                                      stdout=log, stderr=subprocess.STDOUT)
                try:
                    url = f'http://127.0.0.1:{args.port}'
                    wait_ready(url, process=process)
                    if checkpoint_stats(args.model_path) != initial_stats:
                        raise ValueError('checkpoint changed during service loading')
                    actual = json.loads((args.work_root/name/'runtime.json').read_text())
                    if (actual['resolved_image_id'] != runtime['resolved_image_id'] or actual['runtime_status'] != 'passed'
                            or actual['kernel_source_sha256'] != report['source_sha256']
                            or actual.get('host_gpu_allocation')!=allocation.csv):
                        raise ValueError('service image/runtime/allocation differs from passed checkpoint gate')
                    target = native if label == 'native' else out/f'{label}-request.json'
                    result = check_service(url, args.records, target, weave=backend == 'weave_functional',
                        graph=label == 'weave_graph', evidence_port=args.evidence_port_base,
                        native=native if backend == 'weave_functional' else None,
                        expected_source=report['source_sha256'], expected_kernel=args.kernel)
                    report['steps'][label] = {'status': result['status'], 'evidence': str(target)}
                finally:
                    child.stop()
                child.check_cancelled()
                lifecycle = json.loads((args.work_root/name/'container_lifecycle.json').read_text())
                report['steps'][label]['cleanup'] = lifecycle
                if lifecycle['cleanup_status'] not in ('absent', 'stopped'):
                    raise ValueError(f'service container cleanup not demonstrated: {lifecycle}')
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            child.stop()  # Also covers cancellation between spawn and the inner try.
        except BaseException as exc:
            report['status'], report['cleanup_error'] = 'failed', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            if args.controller_lifecycle:
                args.controller_lifecycle.parent.mkdir(parents=True,exist_ok=True)
                args.controller_lifecycle.write_text(json.dumps({
                    'cleanup_status':'not_required' if child.process is None else 'unknown',
                    'controller_status':report['status'],'evidence':str(report_path)})+'\n')
    print(json.dumps({'status': report['status'], 'evidence': str(report_path)}), flush=True)


if __name__ == '__main__':
    main()
