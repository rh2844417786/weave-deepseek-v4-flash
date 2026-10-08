"""Own a foreground Docker client and explicitly clean its container on cancel."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from datetime import datetime, timezone


def _attempt(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                start_new_session=True)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout[-500:], 'stderr': result.stderr[-500:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'command': command, 'error': f'{type(exc).__name__}: {exc}'}


def _inspect(name):
    result = _attempt(['docker', 'inspect', '--format', '{{.State.Running}}', name], 1)
    if result.get('returncode') == 0:
        state = result.get('stdout', '').strip().lower()
        return ('running' if state == 'true' else 'stopped' if state == 'false' else 'unknown'), result
    error = result.get('stderr', '').lower()
    return ('absent' if 'no such object' in error or 'no such container' in error else 'unknown'), result


def cleanup_container(name):
    # GPUQ gives the managed process group ten seconds before SIGKILL. Keep
    # daemon operations bounded and put their CLI helpers in separate sessions.
    attempts = [_attempt(['docker', 'stop', '--timeout', '2', name], 3)]
    if attempts[0].get('returncode') != 0:
        attempts.append(_attempt(['docker', 'rm', '--force', name], 2))
    state, inspection = _inspect(name)
    attempts.append(inspection)
    if state == 'running':
        attempts.append(_attempt(['docker', 'rm', '--force', name], 2))
        state, inspection = _inspect(name)
        attempts.append(inspection)
    return state, attempts


def run_container(command, name, evidence_path, *, stdout=None, exit_code_path=None, expired=None):
    """Manage only the uniquely named container in this exact docker-run argv.

    Signals only set a flag. Popen is allowed to finish registering its child
    before cleanup starts, avoiding cancellation during spawn losing the PID.
    The Docker client has its own session; GPUQ terminates this manager, which
    stops the daemon-owned container and then terminates its own client.
    """
    if not re.fullmatch(r'weave-[A-Za-z0-9_.-]+', name) or command[:2] != ['docker', 'run']:
        raise ValueError('expected an owned weave-* container name and docker run')
    if (command.count('--name') != 1 or command.index('--name')+1 >= len(command)
            or command[command.index('--name')+1] != name or '--rm' not in command):
        raise ValueError('docker container name must match the owned cleanup target and use --rm')
    requested = None

    def remember(signum, frame):
        nonlocal requested
        if requested is None:
            requested = signum

    previous = {sig: signal.signal(sig, remember) for sig in (signal.SIGTERM, signal.SIGINT)}
    process, code, exception = None, None, None
    report = {'container_name': name, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'termination_signal': None, 'exit_code': None, 'cleanup_status': 'not_required'}
    try:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while True:
            if requested is not None:
                code = 128 + requested
                break
            if expired is not None and expired():
                code=process.poll()
                if code is None:
                    code=124
                    report['deadline_expired']=True
                break
            try:
                code = process.wait(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                pass
    except BaseException as exc:
        exception = exc
        code = 1
    finally:
        if requested is not None:
            code = 128 + requested
        if requested is not None or exception is not None or code != 0:
            report['cleanup_status'], report['cleanup_attempts'] = cleanup_container(name)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        report['docker_client_cleanup'] = 'did_not_exit_after_sigkill'
                except ProcessLookupError:
                    pass
        report.update({'termination_signal': requested, 'exit_code': code,
                       'finished_utc': datetime.now(timezone.utc).isoformat()})
        if exception is not None:
            report['launcher_error'] = f'{type(exception).__name__}: {exception}'
        path = Path(evidence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
        if exit_code_path is not None:
            Path(exit_code_path).write_text(str(code)+'\n')
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if exception is not None:
        raise exception
    return report
