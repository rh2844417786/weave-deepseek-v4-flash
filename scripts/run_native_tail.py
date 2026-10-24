#!/usr/bin/env python3
"""Offline native-tail foundations, measured validation and two 1200-second services.

Preview is CPU-only. Execute inside the existing GPUQ allocation with the existing
SGLang image and local read-only checkpoint. No installation or download occurs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_gpqa_c32 import (SECONDS, baseline_coverage, client_command,
    measurement_deadline, validate_runtime, _safe)
from scripts.run_c32_comparison import serving_source_manifest
from scripts.run_model_acceptance import OwnedChild, assert_ports_free
from scripts.run_native_cost import copy_service_evidence
from scripts.check_sglang_service import wait_ready, http_json
from src.weave.calibration import atomic_json
from src.weave.c32_comparison import digest, scheduler_counts
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.gpu_allocation import allocated_gpus
from src.weave.gpqa_c32 import load_gpqa_workload, validate_client_summary, compare_groups
from src.weave.graph_evidence import validate_graph_delta
from src.weave.native_p0_evidence import native_checkpoint_proof
from src.weave.ops import kernel_source_manifest
from src.weave.performance import flush_cache_when_idle
from src.weave.runtime_contract import load_contract
from src.weave.native_tail_cost import METHOD, POLICY, MAX_PLANS, load_profile

GROUPS = ('native', 'peer_tail')
GPU_STAGES = ('compute', 'transport', 'executor', 'executor_memcheck', 'measured_validation')
MEMCHECK_REPORT_API_ERRORS = 'no'
GENERATE = 'python3 scripts/run_native_tail.py --work-root /srv/weave-work --records PREPARED_MAIN --execute'


def safe_tail_public(value):
    """Public artifacts contain aggregates, identities and digests only."""
    _safe(value)
    if isinstance(value, dict):
        forbidden = {'weights', 'routes', 'raw_routes', 'raw_inputs',
                     'sorted_token_ids', 'expert_ids', 'topk_ids', 'topk_weights',
                     'trace', 'traces', 'timeline', 'events', 'task_smids', 'visits',
                     'task_bitmap', 'task_records', 'raw_grouping', 'private'}
        if forbidden.intersection(value) or ('activations' in value and not isinstance(value['activations'], str)):
            raise ValueError('private_tail_payload_cannot_be_exported')
        for child in value.values():
            safe_tail_public(child)
    elif isinstance(value, list):
        for child in value:
            safe_tail_public(child)


def service_command(variant, args, run_name, checkpoint):
    if variant not in (*GROUPS, 'calibration'):
        raise ValueError('unknown native tail service arm')
    command = [sys.executable, str(ROOT/'scripts/run_native_ep.py'), '--execute',
        '--backend', 'native', '--decode-graph', 'full', '--graph-c32', '--request-timing',
        '--work-root', str(args.work_root), '--model-path', str(args.model_path),
        '--run-name', run_name, '--port', str(args.port),
        '--evidence-port-base', str(args.evidence_port_base),
        '--graph-evidence-port-base', str(args.graph_evidence_port_base),
        '--checkpoint-content-sha256', checkpoint]
    if variant != 'native':
        mode = 'calibration' if variant == 'calibration' else 'measured'
        command += ['--native-tail', mode, '--native-tail-directory', str(args.table_dir),
                    '--native-tail-repeats', str(args.native_tail_repeats)]
        if mode == 'calibration':
            command += ['--native-tail-calibration-timeout', str(math.ceil(args.calibration_timeout))]
    return command


def gpu_command(stage, args, image, folder, allocation=None, checkpoint=None):
    if stage not in GPU_STAGES:
        raise ValueError('unknown native tail GPU stage')
    relative = folder.relative_to(args.work_root)
    command = ['docker', 'run', '--rm', '--pull=never', '--name', 'weave-'+folder.parent.name+'-'+stage,
        '--gpus', allocation.docker_argument if allocation else '<GPUQ allocation>',
        '--network=host', '--shm-size=8g', '-v', f'{ROOT}:/workspace:ro',
        '-v', f'{args.work_root}:/work', '-v', f'{args.model_path}:/model:ro', '-w', '/workspace',
        '-e', 'CUDA_VISIBLE_DEVICES='+(allocation.container_visible if allocation else '<container logical devices>'),
        '-e', 'WEAVE_HOST_ALLOCATION='+(allocation.csv if allocation else '<GPUQ allocation>'),
        '-e', f'WEAVE_IMAGE_ID={image}', '-e', f'WEAVE_CHECKPOINT_CONTENT_SHA256={checkpoint or "0"*64}',
        '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'NCCL_SOCKET_IFNAME=lo', '-e', 'NCCL_IB_DISABLE=1',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_DATASETS_OFFLINE=1',
        '-e', 'PYTHONPATH=/workspace/src:/workspace', '-e', 'TORCH_CUDA_ARCH_LIST=9.0a', '-e', 'XDG_CACHE_HOME=/work/cache',
        '-e', 'HF_HOME=/work/cache/huggingface', '-e', 'TORCH_EXTENSIONS_DIR=/work/cache/torch_extensions',
        image, load_contract()['python_executable'], '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=4']
    component = 'executor' if stage in ('measured_validation','executor_memcheck') else stage
    command += [f'/workspace/tests/gpu/check_native_tail_{component}.py',
                '--output-dir', f'/work/{relative}']
    if stage == 'transport':
        command += ['--chunks', '1', '--replays', '9', '--timeout-seconds', '20',
                    '--watchdog-seconds', str(args.gpu_timeout)]
    else:
        command += ['--model-path', '/model', '--cache-root', '/work/cache/weave-native-tail',
                    '--timeout-seconds', str(args.gpu_timeout)]
        if stage == 'measured_validation':
            command += ['--table-dir', '/work/'+str(args.table_dir.relative_to(args.work_root)), '--batches', '32']
    if stage == 'executor_memcheck':
        command += ['--batches','32','--sanitizer-smoke']
        split=command.index(image)+1
        check_script=('native_tail_sanitizer=$(command -v compute-sanitizer) || '
            '{ echo "Installed compute-sanitizer is required; use the existing CUDA13 image/tool path, without online installation." >&2; exit 127; }; '
            'exec "$native_tail_sanitizer" "$@"')
        command=command[:split]+['bash','-c',check_script,'native-tail-memcheck','--tool','memcheck',
            '--target-processes','all','--report-api-errors',MEMCHECK_REPORT_API_ERRORS,
            '--error-exitcode','99']+command[split:]
    return command


def memcheck_evidence():
    """Separate the inspected prior API probes from unobservable new counts.

    report-api-errors=no suppresses API diagnostics entirely. The prior 116 is
    never presented as a current-run measurement, and unknown counts are not 0.
    Process failures, including our checked CUDA API exceptions, remain fatal.
    """
    reference = json.loads((ROOT/'configs/native_tail_memcheck_api_reference.json').read_text())
    return dict(verified=False,tool='compute-sanitizer',mode='memcheck',
        target_processes='all',report_api_errors=MEMCHECK_REPORT_API_ERRORS,
        api_reporting_scope='all_CUDA_API_reports',error_exit_code=99,
        actual_process_exit_code=None,application_nonzero_exit_is_hard_failure=True,
        scope='B32_executor_native_down_and_peer_IPC',
        api_error_observations=dict(
            current_run=dict(filtered_count=None,categories=None,
                count_status='not_observable_with_report_api_errors_no'),
            historical_reference=reference))


def export_public(run, destination):
    """Explicit allowlist. GPU raw rank reports remain private; check.json is sanitized."""
    run, destination = Path(run), Path(destination)
    if destination.exists():
        raise FileExistsError('fresh_public_destination_required')
    names = ['experiment.json', 'comparison.json', 'source-manifest.json', 'serving-source-manifest.json',
             'tail-source-manifest.json', 'data-manifest.json', 'native-checkpoint-proof.json',
             'calibration/check.json', 'calibration/tail-ready.json', 'calibration/launch.json',
             'calibration/runtime.json', 'calibration/server-lifecycle.json', 'calibration/tail-cleanup.json',
             'calibration/performance-analysis.json']
    names += [f'{stage}/{name}' for stage in GPU_STAGES for name in ('check.json', 'container-lifecycle.json')]
    names += [f'calibration/native-tail-rank{r}.json' for r in range(4)]
    group_names = ('group.json', 'launch.json', 'runtime.json', 'server-lifecycle.json',
        'measurement-lifecycle.json', 'client-summary.json', 'measurement-summary.json',
        'gpqa-rows.json', 'load-window.json', 'load-progress.json', 'warmup-ready.json',
        'scheduler-summary.json', 'graph-before.json', 'graph-after.json', 'graph-check.json',
        'tail-before.json', 'tail-after.json', 'tail-check.json', 'tail-cleanup.json')
    names += [f'{group}/{name}' for group in GROUPS for name in group_names]
    files = {}
    for name in names:
        path = run/name
        # Reject symlinked parents as well as a symlinked leaf.
        for parent in (path, *path.parents):
            if parent == run.parent:
                break
            if parent.is_symlink():
                raise ValueError('public_symlink_forbidden')
        if path.is_file():
            raw = path.read_bytes(); safe_tail_public(json.loads(raw)); files[name] = raw
    if 'experiment.json' not in files:
        raise ValueError('experiment report required for export')
    destination.mkdir(parents=True)
    for name, raw in files.items():
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw); path.chmod(0o664)
    manifest = dict(schema_version=1, status=json.loads(files['experiment.json'])['status'],
                    raw_content_exported=False,
                    files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    atomic_json(destination/'manifest.json', manifest)
    (destination/'manifest.json').chmod(0o664)
    return manifest


def profile_summary(profiles):
    return {rank: {'identity': p['identity'], 'profile_sha256': p['profile_sha256'], 'groups': p['groups']}
            for rank, p in profiles.items()}


def _timing_evidence(value, repeats):
    samples=value.get('samples_us',[]) if isinstance(value,dict) else []
    if (len(samples)!=repeats or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples)
            or value.get('median_us')!=statistics.median(samples)):
        raise ValueError('complete finite repeated CUDA timing evidence required')


def _table_execution_evidence(table):
    measured=table.get('measurements',{});repeats=measured.get('repeats')
    if measured.get('complete') is not True or type(repeats) is not int or repeats<2:
        raise ValueError('complete repeated native tail measurements required; '+GENERATE)
    anchors=measured.get('anchors',[]);plans=table['plans']
    if len(anchors)!=4 or [a.get('index') for a in anchors]!=[0,1,4,5]:
        raise ValueError('four observed training anchors required')
    for anchor in anchors:
        candidates=anchor.get('candidates',[])
        if len(candidates)!=len(plans) or [c.get('plan') for c in candidates]!=plans:
            raise ValueError('complete native tail candidate measurement coverage required')
        for candidate in candidates:
            _timing_evidence(candidate.get('full_moe',{}),repeats)
            observations=candidate.get('diagnostic_observations',[])
            if len(observations)!=repeats or any(v.get('protocol_valid') is not True for v in observations):
                raise ValueError('each repeated diagnostic needs valid protocol evidence')
    heldout=table.get('heldout',[])
    if len(heldout)!=2 or [v.get('index') for v in heldout]!=[6,7]:
        raise ValueError('two independent held-out executions required per tail table')
    for case in heldout:
        numeric=case.get('native_numerical',{})
        if case.get('cpu_gpu_selection_agree') is not True or numeric.get('finite') is not True:
            raise ValueError('held-out GPU planner validity and finite output required')
        for key in ('relative_l2','max_abs'):
            value=numeric.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('finite numerical records required')
        _timing_evidence(case.get('native_full_moe',{}),repeats)
        _timing_evidence(case.get('selected_full_moe',{}),repeats)
        fixed=case.get('fixed_candidates',[])
        if len(fixed)!=len(plans) or [v.get('plan') for v in fixed]!=plans:
            raise ValueError('complete held-out candidate measurements required')
        for value in fixed:_timing_evidence(value.get('full_moe',{}),repeats)


def load_profiles(directory, identity):
    expected = dict(tail_source_sha256=identity['native_tail_source_sha256'],
                    upstream_sha256=identity['native_tail_upstream_sha256'], image_id=identity['image_id'],
                    checkpoint_content_sha256=identity['checkpoint_content_sha256'])
    profiles = {}
    for rank in range(4):
        path = Path(directory)/f'native-tail-rank{rank}.json'
        p = load_profile(path, expected_identity=dict(expected, rank=rank))
        safe_tail_public(p)
        groups = p.get('groups', {})
        layers = [layer for group in groups.values() for layer in group.get('layers', [])]
        if sorted(layers) != list(range(43)) or any(type(v) is not int for v in layers):
            raise ValueError('complete unique structural grouping of all 43 layers required; '+GENERATE)
        expected_tables = {f'{g}:{batch}' for g in groups for batch in range(1, 33)}
        if set(p['tables']) != expected_tables:
            raise ValueError('complete measured group/batch tables for exact full graph batches 1..32 required; '+GENERATE)
        for key, table in p['tables'].items():
            if table['total_rows'] != int(key.rsplit(':', 1)[1])*6:
                raise ValueError('tail table routed capacity differs from native top-k geometry')
            _table_execution_evidence(table)
        profiles[str(rank)] = p
    first = profiles['0']
    for p in profiles.values():
        if p['groups'] != first['groups'] or set(p['tables']) != set(first['tables']):
            raise ValueError('structural group/table mapping differs across ranks')
        for key in ('runtime', 'sglang_commit', 'model_config_sha256', 'graph_contract'):
            if key in first['identity'] and p['identity'].get(key) != first['identity'][key]:
                raise ValueError('cross-rank deployment identity differs: '+key)
    return profiles


def _index_tail(snapshots, identity, mode):
    if len(snapshots) != 4 or {s.get('rank') for s in snapshots} != set(range(4)):
        raise ValueError('four distinct native tail ranks required')
    indexed = {}
    hashes = {r: p['profile_sha256'] for r, p in identity['tail_profiles'].items()}
    for snapshot in snapshots:
        rank = snapshot['rank']; p = identity['tail_profiles'][str(rank)]
        ex = snapshot.get('executor', {})
        if (snapshot.get('schema_version') != 1 or snapshot.get('method') != METHOD
                or snapshot.get('mode') != mode or snapshot.get('layer_count') != 43
                or snapshot.get('numerical_policy') != 'record_only'
                or snapshot.get('source_sha256') != identity['native_tail_source_sha256']
                or snapshot.get('upstream_sha256') != identity['native_tail_upstream_sha256']
                or snapshot.get('groups') != p['groups']
                or ex.get('schema_version') != 1 or ex.get('rank') != rank or ex.get('mode') != mode
                or ex.get('source_sha256') != identity['native_tail_source_sha256']
                or ex.get('profile_sha256') != p['profile_sha256']
                or ex.get('profiles_sha256') != hashes or ex.get('identity') != p['identity']
                or ex.get('calibration_complete') is not True):
            raise ValueError('native tail source/runtime/measured-table identity differs')
        rows = ex.get('rows', [])
        if len(rows) != 43 or {r.get('layer') for r in rows} != set(range(43)):
            raise ValueError('complete unique 43-layer gate/up and tail coverage required')
        for row in rows:
            values = [row.get(k) for k in ('gate_up_calls', 'tail_calls', 'committed_calls',
                                          'plan_search_calls', 'empty_calls')]
            histogram = row.get('histogram', [])
            if (any(type(v) is not int or v < 0 for v in values)
                    or len(histogram) != MAX_PLANS or any(type(v) is not int or v < 0 for v in histogram)
                    or sum(histogram) != row['tail_calls'] or row['empty_calls'] > row['tail_calls']
                    or row['committed_calls'] != row['tail_calls']):
                raise ValueError('malformed native tail counters or uncommitted invocation')
        indexed[rank] = {r['layer']: r for r in rows}
    return indexed


def validate_tail_delta(before, after, graph_check, identity):
    old, new = _index_tail(before, identity, 'measured'), _index_tail(after, identity, 'measured')
    replays = graph_check.get('replays')
    if type(replays) is not int or replays <= 0:
        raise ValueError('positive actual graph replay count required')
    result = dict(status='passed', method=METHOD, policy=POLICY, numerical_policy='record_only',
                  replays=replays, total_gate_up_calls=0, total_tail_calls=0,
                  total_plan_search_calls=0, plans_with_variation=0, ranks={})
    for rank in range(4):
        rows = []
        for layer in range(43):
            a, b = old[rank][layer], new[rank][layer]
            deltas = {k: b[k]-a[k] for k in ('gate_up_calls','tail_calls','committed_calls','plan_search_calls','empty_calls')}
            hist = [end-start for start,end in zip(a['histogram'],b['histogram'])]
            if (any(deltas[k] != replays for k in ('gate_up_calls','tail_calls','committed_calls','plan_search_calls'))
                    or not 0 <= deltas['empty_calls'] <= replays or min(hist) < 0 or sum(hist) != replays):
                raise ValueError('one native gate/up, tail, committed protocol and planner required per layer/replay')
            row = dict(layer=layer, histogram=hist, **deltas)
            for field in ('published_chunks', 'down_tiles', 'stolen_tiles', 'remote_read_bytes'):
                if field in a or field in b:
                    if any(type(x.get(field)) is not int or x[field] < 0 for x in (a,b)) or b[field] < a[field]:
                        raise ValueError('invalid aggregate native tail counter: '+field)
                    row[field] = b[field]-a[field]
            rows.append(row)
            result['total_gate_up_calls'] += deltas['gate_up_calls']
            result['total_tail_calls'] += deltas['tail_calls']
            result['total_plan_search_calls'] += deltas['plan_search_calls']
            result['plans_with_variation'] += int(sum(n > 0 for n in hist) > 1)
        result['ranks'][str(rank)] = {'rows':rows}
    result['plan_variation_observed'] = result['plans_with_variation'] > 0
    result['interpretation'] = 'Observed worker plans and stealing counts are records; invariant plans and zero stealing are valid observations.'
    return result


def compare_tail_groups(groups):
    comparison = compare_groups({'native':groups['native'], 'hopper_prefill':groups['peer_tail']})
    rates = comparison['decode_tokens_per_second']
    comparison['peer_tail_vs_native_whole_window_ratio'] = rates['hopper_prefill']/rates['native']
    comparison['peer_tail_vs_native_sustained_c32_ratio'] = comparison.pop('prefill_vs_native_ratio')
    rates['peer_tail'] = rates.pop('hopper_prefill')
    comparison.update(method=METHOD, policy=POLICY, complete_weave_claim=False, initial_concurrency=32,
        group_labels={'native':'Original native SGLang', 'peer_tail':'Native front and resident peer tail'},
        generation_agreement_evaluated=False, numerical_policy='record_only', performance_policy='record_only',
        prefill_implementation='original_native', measurement_mode='native_tail_performance', decode_graph='full')
    return comparison


def close_tail_service(args, folder, server_dir=None):
    """HTTP barrier: all ranks disconnect peers before any frees its allocation.

    Never use the live scheduler's TP CPU group for background cleanup. One
    deadline covers both parallel four-rank phases. Failed prepare means no
    commit; the owning process then tears down the failed container.
    """
    result = dict(status='unverified', protocol='http_two_phase_v1',
        phase='prepare', ipc_closed=False, graphs_reset=None, ranks={})
    deadline = time.monotonic() + args.cleanup_timeout
    fields = ('schema_version', 'protocol', 'rank', 'status', 'phase',
              'graphs_owned', 'graphs_reset', 'peers_disconnected', 'ipc_closed',
              'completed_phases', 'updated_utc', 'error_type')
    def public_progress(reply):
        row = {key:reply[key] for key in fields if key in reply}
        safe_tail_public(row)
        return row
    def phase(endpoint, expected_status):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('native_tail_cleanup_deadline')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {rank:pool.submit(http_json,
                f'http://127.0.0.1:{args.graph_evidence_port_base+rank}/{endpoint}',
                timeout=remaining) for rank in range(4)}
            errors = []
            for rank, future in futures.items():
                try:
                    reply = future.result()
                    if reply.get('rank') != rank:
                        raise ValueError('tail cleanup rank identity differs')
                    result['ranks'][str(rank)] = public_progress(reply)
                    owned, reset = reply.get('graphs_owned'), reply.get('graphs_reset')
                    if (reply.get('status') != expected_status or type(owned) is not int
                            or owned < 1 or type(reset) is not int or reset != owned
                            or reply.get('peers_disconnected') is not True
                            or reply.get('ipc_closed') is not (expected_status == 'complete')):
                        raise ValueError('graph reset and IPC close evidence incomplete')
                except Exception as exc:
                    errors.append(dict(rank=rank, phase=result['phase'], error_type=type(exc).__name__))
            if errors:
                result['errors'] = errors
                result['error_types'] = [error['error_type'] for error in errors]
                raise RuntimeError('native_tail_service_cleanup_unverified')
    try:
        atomic_json(Path(folder)/'tail-cleanup.json', result)
        phase('tail-close-prepare', 'prepared')
        result['phase'] = 'commit'
        atomic_json(Path(folder)/'tail-cleanup.json', result)
        phase('tail-close', 'complete')
        result.update(status='complete', phase='complete', ipc_closed=True)
        return result
    except Exception as exc:
        result.setdefault('error_types', [type(exc).__name__])
        raise
    finally:
        if result['status'] != 'complete' and server_dir is not None:
            # Root-owned container files are atomic 0664 JSON. A journal remains
            # readable even when a CUDA call blocks its HTTP handler/GIL.
            for rank in range(4):
                path = Path(server_dir)/'native-tail'/f'cleanup-rank{rank}.json'
                try:
                    row = json.loads(path.read_text())
                    if row.get('rank') != rank:
                        raise ValueError('cleanup journal rank identity differs')
                    result['ranks'][str(rank)] = public_progress(row)
                except Exception as exc:
                    result.setdefault('progress_read_errors', {})[str(rank)] = type(exc).__name__
        resets = [row.get('graphs_reset') for row in result['ranks'].values()]
        known = [count for count in resets if type(count) is int and count >= 0]
        result['graphs_reset_observed'] = sum(known) if known else None
        result['graphs_reset'] = sum(known) if len(known) == 4 else None
        result['progress_rank_count'] = len(result['ranks'])
        atomic_json(Path(folder)/'tail-cleanup.json', result)


def _unverified_tail_cleanup(folder, reason):
    atomic_json(Path(folder)/'tail-cleanup.json',dict(status='unverified',ipc_closed=False,
        graphs_reset=None,reason=reason,container_cleanup_evidence='server-lifecycle.json'))


def run_group(args, variant, out, run_name, identity, workload, image, env, owner, client_owner, check_cancelled):
    folder = out / variant; folder.mkdir()
    server_dir = args.work_root / run_name; url = f'http://127.0.0.1:{args.port}'
    group = dict(status='running', variant=variant, stage='starting', identity=identity)
    service_ready = False
    ports = [args.port] + list(range(args.evidence_port_base, args.evidence_port_base + 4)) + list(range(args.graph_evidence_port_base, args.graph_evidence_port_base + 4))
    try:
        assert_ports_free(ports)
        with (folder / 'server-launch-private.log').open('w') as log:
            server = owner.spawn(service_command(variant, args, run_name, identity['checkpoint_content_sha256']),
                server_dir / 'container_lifecycle.json', env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(url, timeout=args.setup_timeout, process=server)
            service_ready = True
            validate_runtime(json.loads((server_dir / 'runtime.json').read_text()), identity, env['CUDA_VISIBLE_DEVICES'])
            flush_cache_when_idle(url, deadline=time.monotonic() + args.drain_timeout, max_wait=args.drain_timeout)
            command = client_command(image, args.work_root, args.records, f'{out.name}/{variant}', out.name + '-' + variant, args.port)
            lifecycle = folder / 'measurement-lifecycle.json'
            manager = [sys.executable, str(ROOT / 'scripts/manage_container.py'), '--name', command[command.index('--name') + 1],
                       '--evidence', str(lifecycle), '--log', str(folder / 'client-private.log'), '--'] + command
            child = client_owner.spawn(manager, lifecycle, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            started = time.monotonic(); deadline = started + args.warmup_timeout; marker = None; baseline = False
            while child.poll() is None:
                check_cancelled()
                if server.poll() is not None:
                    raise RuntimeError('service_exited_during_client')
                if not baseline and (folder / 'warmup-ready.json').is_file():
                    ready = json.loads((folder / 'warmup-ready.json').read_text()).get('ready_monotonic')
                    now = time.monotonic()
                    if type(ready) not in (float, int) or not math.isfinite(ready) or ready > now:
                        raise ValueError('invalid_warmup_ready_marker')
                    idle_deadline = min(now + 20., ready + 20.)
                    graph_before = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='graph-coverage')
                    atomic_json(folder / 'graph-before.json', graph_before)
                    if variant == 'peer_tail':
                        tail_before = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='tail-coverage')
                        atomic_json(folder / 'tail-before.json', tail_before)
                    log_offset = (server_dir / 'server.log').stat().st_size
                    if time.monotonic() >= idle_deadline:
                        raise TimeoutError('idle_baseline_deadline')
                    atomic_json(folder / 'measurement-go.json', {'status': 'baseline_captured'})
                    baseline = True; group['stage'] = 'measurement'; atomic_json(folder / 'group.json', group)
                if (folder / 'load-window.json').is_file():
                    current = json.loads((folder / 'load-window.json').read_text())
                    if not baseline or (marker is not None and marker != current):
                        raise ValueError('measurement_window_changed_or_preceded_baseline')
                    marker = current
                    deadline = measurement_deadline(marker, started, time.monotonic(), args.drain_timeout + 30.)
                if time.monotonic() >= deadline:
                    raise TimeoutError('client_measurement_deadline' if marker else 'client_warmup_deadline')
                time.sleep(.2)
            if child.returncode != 0:
                raise RuntimeError('measurement_client_failed')
            client_owner.stop(); check_cancelled()
            if marker is None or not baseline:
                raise ValueError('missing_measurement_window')
            client = validate_client_summary(json.loads((folder / 'measurement-summary.json').read_text()))
            if client.get('workload') != workload.identity:
                raise ValueError('client_workload_identity_differs')
            atomic_json(folder / 'client-summary.json', client)
            flush_cache_when_idle(url, deadline=time.monotonic() + args.drain_timeout, max_wait=args.drain_timeout)
            with (server_dir / 'server.log').open('rb') as stream:
                stream.seek(log_offset); logs = stream.read().decode(errors='replace')
            atomic_json(folder / 'scheduler-summary.json', scheduler_counts(logs))
            idle_deadline = time.monotonic() + 20.
            graph_after = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='graph-coverage')
            atomic_json(folder / 'graph-after.json', graph_after)
            graph_check = validate_graph_delta(graph_before, graph_after,
                expected_observer_source=identity['observer_source_sha256'], expected_sources=identity['graph_sources'])
            atomic_json(folder / 'graph-check.json', graph_check)
            if variant == 'peer_tail':
                tail_after = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='tail-coverage')
                atomic_json(folder / 'tail-after.json', tail_after)
                tail_check = validate_tail_delta(tail_before, tail_after, graph_check, identity)
                atomic_json(folder / 'tail-check.json', tail_check); group['tail_evidence'] = tail_check
            if variant == 'peer_tail':
                group['tail_cleanup'] = close_tail_service(args,folder,server_dir=server_dir)
            group.update(status='passed', stage='complete', client_summary=client, batch_evidence=graph_check)
        owner.stop(); check_cancelled()
    except BaseException as exc:
        group.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        if variant == 'peer_tail' and not (folder/'tail-cleanup.json').exists():
            if service_ready:
                try: close_tail_service(args,folder,server_dir=server_dir)
                except BaseException as exc: group.update(status='failed',cleanup_error_type=type(exc).__name__)
            else:
                _unverified_tail_cleanup(folder,'service_not_ready_or_aborted')
        atomic_json(folder / 'group.json', group)
        copy_service_evidence(server_dir, folder)
    return group


def calibration_ready(snapshots, profiles, identity):
    bound = dict(identity, tail_profiles=profile_summary(profiles))
    _index_tail(snapshots, bound, 'calibration')


def run_calibration(args, out, run_name, identity, env, owner, check_cancelled):
    folder = out/'calibration'; folder.mkdir(exist_ok=True)
    server_dir = args.work_root/run_name
    result = dict(status='running', mode='calibration', numerical_policy='record_only')
    service_ready = False
    try:
        assert_ports_free([args.port]+list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4)))
        with (folder/'server-launch-private.log').open('w') as log:
            server = owner.spawn(service_command('calibration',args,run_name,identity['checkpoint_content_sha256']),
                server_dir/'container_lifecycle.json', env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(f'http://127.0.0.1:{args.port}',timeout=args.calibration_timeout,process=server)
            service_ready = True
            check_cancelled()
            validate_runtime(json.loads((server_dir/'runtime.json').read_text()),identity,env['CUDA_VISIBLE_DEVICES'])
            profiles = load_profiles(args.table_dir,identity)
            snapshots = baseline_coverage(args.graph_evidence_port_base,deadline=time.monotonic()+60.,endpoint='tail-coverage')
            calibration_ready(snapshots,profiles,identity)
            atomic_json(folder/'tail-ready.json',snapshots)
            result.update(status='complete', service_ready=True, profiles=profile_summary(profiles),
                          calibration_complete=True, heldout_validation='complete_record_only')
        return profiles
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        try:
            if service_ready:
                close_tail_service(args,folder,server_dir=server_dir)
            else:
                _unverified_tail_cleanup(folder,'service_not_ready_or_aborted')
        except BaseException as exc:
            result.update(status='failed',cleanup_error_type=type(exc).__name__)
            raise
        finally:
            try:
                owner.stop()
            except BaseException as exc:
                result.update(status='failed',container_cleanup_error_type=type(exc).__name__)
                raise
            finally:
                copy_service_evidence(server_dir,folder)
                atomic_json(folder/'check.json',result)


def summarize_gpu_reports(stage, folder, identity):
    """Validate raw private reports, then select public aggregate evidence explicitly."""
    reports = {}
    expected_compute = {(b,m,p) for b in (32,65,128) for m in (8,16,32,48,64)
                        for p in ('concentrated','balanced','light','empty')}
    for rank in range(4):
        component = 'executor' if stage in ('measured_validation','executor_memcheck') else stage
        name = f'rank{rank}.json' if stage == 'transport' else f'native-tail-{component}-rank{rank}.json'
        path = folder/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('complete four-rank GPU reports required: '+stage)
        raw = path.read_bytes(); report = json.loads(raw)
        if (report.get('rank') != rank or report.get('method') != METHOD
                or report.get('numerical_policy') != 'record_only'):
            raise ValueError('GPU report rank/method/policy mismatch: '+stage)
        cases = report.get('cases',[])
        if stage == 'transport':
            graph = report.get('graph_capture_replay',{})
            if (report.get('valid') is not True or report.get('cleanup') is not True
                    or report.get('gpu_mechanisms_verified') is not True
                    or graph.get('verified') is not True or graph.get('replays',0)<5
                    or graph.get('final_sequence',0)-graph.get('initial_sequence',0)!=graph['replays']
                    or not cases or any(c.get('valid') is not True for c in cases)):
                raise ValueError('complete IPC, graph replay, taskset and cleanup evidence required')
            expected_names = {'all_inputs_ready','delayed_rank_3','empty_local_contribution_rank_2',
                              'zero_token_payload','timeout','gpu_cancel','identity_mismatch','nonfinite'}
            expected_names |= {f'graph_replay_{i}' for i in range(graph['replays'])}
            if {c.get('name') for c in cases} != expected_names or len(cases)!=len(expected_names):
                raise ValueError('transport controlled-case coverage differs')
            from src.weave.native_tail_transport import source_manifest
            manifest=source_manifest();manifest['cuda_arch_list']='9.0a'
            expected_source = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if report.get('source_sha256') != expected_source:
                raise ValueError('transport compiled-source identity differs')
        else:
            if (report.get('status')!='complete' or report.get('checks_complete') is not True
                    or report.get('cleanup',{}).get('status')!='complete' or not cases):
                raise ValueError('complete GPU compute/protocol checks and cleanup required: '+stage)
            if (report.get('runtime',{}).get('image_id') != identity['image_id']
                    or report.get('upstream_sha256') != identity['native_tail_upstream_sha256']):
                raise ValueError('GPU runtime or upstream source identity differs')
            if report.get('case_count') != len(cases):
                raise ValueError('GPU case count differs from evidence')
            if stage == 'compute':
                if {(c.get('batch'),c.get('m_block'),c.get('pattern')) for c in cases} != expected_compute or len(cases)!=len(expected_compute):
                    raise ValueError('all native M blocks, workloads and independent batch checks required')
                if report.get('upstream_sha256') != identity['native_tail_upstream_sha256']:
                    raise ValueError('S1 upstream source identity differs')
                for c in cases:
                    if (c.get('exact_task_bitmap_verified') is not True or c.get('guards_intact') is not True
                            or c.get('stable_graph_pointers') is not True or c.get('replay_cycles',0)<2):
                        raise ValueError('S1 exact compute coverage and graph evidence required')
            else:
                graph=report.get('graph_capture_replay',{})
                if graph.get('verified') is not True or graph.get('replays',0)<5:
                    raise ValueError('executor actual cooperative graph replay evidence required')
                if report.get('expected_case_count') != len(cases):
                    raise ValueError('executor expected case set incomplete')
                if (report.get('source_sha256') != identity['native_tail_source_sha256']
                        or report.get('harness_sha256') != hashlib.sha256((ROOT/'tests/gpu/check_native_tail_executor.py').read_bytes()).hexdigest()):
                    raise ValueError('executor source identity differs')
                batches=(32,65) if stage=='executor' else (32,)
                modes=('measured',) if stage=='measured_validation' else ('serial','C1','C2','steal_uninjected','steal_controlled')
                expected={(b,m,p,i) for b in batches for m in modes
                          for p in ('concentrated','balanced','light','empty_rank2') for i in range(5)}
                actual={(c.get('batch'),c.get('mode'),c.get('pattern'),c.get('replay')) for c in cases}
                if actual!=expected or len(cases)!=len(expected):
                    raise ValueError('executor exact mechanism/workload/replay case matrix differs')
                for c in cases:
                    if (any(c.get(k) is not True for k in ('protocol_valid','canary_ok','native_locks_zero',
                            'stable_graph_addresses','exact_task_coverage','finalize_task_coverage_verified')) or c.get('front_and_tail_calls')!=1):
                        raise ValueError('executor per-case taskset, protocol, memory or capture evidence invalid')
                if stage == 'measured_validation':
                    if (report.get('profile_sha256') != identity['tail_profiles'][str(rank)]['profile_sha256']
                            or report.get('planner_validation',{}).get('complete') is not True):
                        raise ValueError('measured GPU planner validity/profile binding required')
        row = {'status':'complete','rank':rank,'case_count':len(cases),'cleanup':'complete',
               'report_sha256':hashlib.sha256(raw).hexdigest(),
               'gpu_mechanisms_verified':report.get('gpu_mechanisms_verified') is True}
        # These are aggregate fields; task vectors, timelines and raw rank reports stay private.
        for key in ('graph_capture_replay','profile_sha256','planner_validation',
                    'finalize_task_coverage_verified','execution_timing_summary',
                    'overlap_observed','controlled_stealing_observed',
                    'controlled_stolen_worker_resumed_communication','uninjected_operator_stealing_observed'):
            if key in report:
                row[key]=report[key]
        row['numerical_policy']='record_only'
        safe_tail_public(row)
        reports[str(rank)] = row
    return dict(status='complete',method=METHOD,stage=stage,numerical_policy='record_only',ranks=reports,
                gpu_mechanisms_verified=all(r['gpu_mechanisms_verified'] for r in reports.values()))


def run_gpu_stage(stage,args,out,identity,image,allocation,owner,check_cancelled):
    folder=out/stage;folder.mkdir()
    command=gpu_command(stage,args,image,folder,allocation,identity['checkpoint_content_sha256'])
    lifecycle=folder/'container-lifecycle.json'
    manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
             '--evidence',str(lifecycle),'--log',str(folder/'gpu-private.log'),'--']+command
    result=dict(status='running',method=METHOD,stage=stage,numerical_policy='record_only')
    memory_check=memcheck_evidence() if stage=='executor_memcheck' else None
    if memory_check is not None:
        result['memory_check']=memory_check
    try:
        child=owner.spawn(manager,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+args.gpu_timeout+120.
        while child.poll() is None:
            check_cancelled()
            if time.monotonic()>=deadline:
                raise TimeoutError('native_tail_gpu_stage_deadline')
            time.sleep(.2)
        if memory_check is not None:
            memory_check['actual_process_exit_code']=child.returncode
        if child.returncode!=0:
            raise RuntimeError('native_tail_gpu_stage_failed')
        owner.stop();check_cancelled()
        result=summarize_gpu_reports(stage,folder,identity)
        if memory_check is not None:
            memory_check['verified']=True
            result['memory_check']=memory_check
        return result
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(folder/'check.json',result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--records',type=Path)
    parser.add_argument('--stage',choices=('foundations','all'),default='all')
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--native-baseline-dir',type=Path,default=ROOT/'reports/server/gpqa-c32-9f1635df2afe')
    parser.add_argument('--calibration-dir',type=Path,help='explicit reuse of complete matching measured profiles under WORK_ROOT')
    parser.add_argument('--native-tail-repeats',type=int,default=7)
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--calibration-timeout',type=float,default=43200.)
    parser.add_argument('--gpu-timeout',type=int,default=28800)
    parser.add_argument('--setup-timeout',type=float,default=28800.)
    parser.add_argument('--warmup-timeout',type=float,default=960.)
    parser.add_argument('--drain-timeout',type=float,default=60.)
    parser.add_argument('--cleanup-timeout',type=float,default=300.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--graph-evidence-port-base',type=int,default=30310)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.stage=='all' and args.records is None:
        parser.error('--records required for full calibration and formal comparison')
    if args.native_tail_repeats<2:
        parser.error('native tail requires at least two actual repeats')
    if args.stage=='foundations' and args.calibration_dir:
        parser.error('--calibration-dir applies to --stage all')
    for key in ('work_root','records','model_path','native_baseline_dir','export_root','calibration_dir'):
        path=getattr(args,key)
        if path is not None:
            if not path.is_absolute() or '..' in path.parts:
                parser.error(key+' must be absolute without traversal')
            setattr(args,key,path.resolve())
    if args.calibration_dir:
        try:
            args.calibration_dir.relative_to(args.work_root)
        except ValueError:
            parser.error('calibration-dir must be under WORK_ROOT')
    if any(not math.isfinite(v) or v<=0 for v in (args.calibration_timeout,args.setup_timeout,args.warmup_timeout,args.drain_timeout,args.cleanup_timeout)) or args.gpu_timeout<30:
        parser.error('positive finite timeouts and GPU timeout >=30 seconds required')
    ports=[args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4))+list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4))
    if min(ports)<1 or max(ports)>65535 or len(set(ports))!=len(ports):
        parser.error('valid distinct serving/evidence ports required')
    name='native-tail-'+uuid.uuid4().hex[:12]
    out=args.work_root/name;destination=(args.export_root or args.work_root/'public-reports')/name
    args.table_dir=args.calibration_dir or out/'calibration'
    contract=load_contract()
    stages=['compute','transport']
    if args.stage=='all':
        stages += ['executor','executor_memcheck','reuse_calibration' if args.calibration_dir else 'calibration','measured_validation',*GROUPS]
    if not args.execute:
        preview=dict(status='preview_not_executed',method=METHOD,policy=POLICY,numerical_policy='record_only',
            stages=stages,gpu_commands={stage:gpu_command(stage,args,contract['image'],out/stage) for stage in stages if stage in GPU_STAGES},
            calibration=service_command('calibration',args,'native-ep-'+name+'-calibration','0'*64) if args.stage=='all' and not args.calibration_dir else None,
            groups={variant:service_command(variant,args,'native-ep-'+name+'-'+variant,'0'*64) for variant in GROUPS} if args.stage=='all' else {},
            clients={variant:client_command(contract['image'],args.work_root,args.records,f'{name}/{variant}',name+'-'+variant,args.port) for variant in GROUPS} if args.stage=='all' else {},
            seconds_per_arm=SECONDS,initial_concurrency=32,captured_batches=list(range(1,33)),
            prefill_implementation='original_native',decode_graph='full',complete_weave_claim=False,
            calibration_repeats=args.native_tail_repeats,calibration_directory=str(args.table_dir),
            required_environment={'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'},
            evidence_reads='idle_before_and_after_only',source_implemented=True,gpu_mechanisms_verified=False,
            full_model_graph_execution_verified=False,throughput_result_observed=False)
        print(json.dumps(preview,indent=2));return 0
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False)
    owner,client_owner=OwnedChild(),OwnedChild()
    report=dict(schema_version=1,status='running',stage='validating',method=METHOD,policy=POLICY,
        numerical_policy='record_only',generation_comparison_policy='record_only',generation_agreement_evaluated=False,
        complete_weave_claim=False,initial_concurrency=32,seconds_per_arm=SECONDS,groups={},gpu_checks={},
        source_implemented=True,gpu_mechanisms_verified=False,full_model_graph_execution_verified=False,
        throughput_result_observed=False,requested_stage=args.stage,
        started_utc=datetime.now(timezone.utc).isoformat(),gpuq_job_id=os.environ.get('GPUQ_JOB_ID'),host_gpu_allocation=allocation.csv)
    code=1
    def save():atomic_json(out/'experiment.json',report)
    def cancel(number,frame):
        for child in (owner,client_owner):child.requested=number
        if not any(child.protected for child in (owner,client_owner)):
            raise InterruptedError('job_cancelled')
    def check_cancelled():owner.check_cancelled();client_owner.check_cancelled()
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        save()
        workload=load_gpqa_workload(args.records,model_path=args.model_path) if args.stage=='all' else None
        source=serving_source_manifest();source_digest=digest(source)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        if not image.startswith('sha256:'):
            raise ValueError('content-addressed existing local image required')
        proof=native_checkpoint_proof(args.native_baseline_dir,args.model_path)
        from src.weave.native_tail_runtime import native_tail_source_sha256
        from src.weave.graph_hooks import graph_observer_source_sha256
        pins=json.loads((ROOT/'configs/native_tail_sources.json').read_text())['sha256']
        identity=dict(source_sha256=digest(kernel_source_manifest()),checkpoint_content_sha256=proof['checkpoint_content_sha256'],
            image_id=image,serving_source_sha256=source_digest,decode_graph='full',
            native_tail_source_sha256=native_tail_source_sha256(),native_tail_upstream_sha256=pins,
            observer_source_sha256=graph_observer_source_sha256(),graph_sources=json.loads((ROOT/'configs/sglang_graph_sources.json').read_text())['sha256'])
        if workload is not None:identity.update(workload.identity)
        report['identity']=identity;save()
        for file,value in (('native-checkpoint-proof.json',proof),('source-manifest.json',kernel_source_manifest()),
                           ('serving-source-manifest.json',source),('tail-source-manifest.json',
                            dict(source_sha256=identity['native_tail_source_sha256'],upstream_sha256=pins))):
            atomic_json(out/file,value)
        if workload is not None:atomic_json(out/'data-manifest.json',workload.identity)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
             'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        for stage in ('compute','transport')+ (('executor','executor_memcheck') if args.stage=='all' else ()):
            check_cancelled();report['stage']=stage;save()
            report['gpu_checks'][stage]=run_gpu_stage(stage,args,out,identity,image,allocation,owner,check_cancelled);save()
        if args.stage=='foundations':
            report.update(status='foundations_complete',stage='complete',
                          gpu_mechanisms_verified=all(v['gpu_mechanisms_verified'] for v in report['gpu_checks'].values()))
            code=0
        else:
            check_cancelled();report['stage']='calibration';save()
            if args.calibration_dir:
                profiles=load_profiles(args.table_dir,identity)
                folder=out/'calibration';folder.mkdir()
                for rank,profile in profiles.items():atomic_json(folder/f'native-tail-rank{rank}.json',profile)
                atomic_json(folder/'check.json',dict(status='complete',mode='explicit_profile_reuse',
                    profiles=profile_summary(profiles),heldout_validation='complete_record_only',numerical_policy='record_only'))
            else:
                profiles=run_calibration(args,out,'native-ep-'+name+'-calibration',identity,env,owner,check_cancelled)
            from src.weave.native_tail_timing import summarize_profiles
            atomic_json(out/'calibration/performance-analysis.json',summarize_profiles(profiles))
            identity['tail_profiles']=profile_summary(profiles);save()
            check_cancelled();report['stage']='measured_validation';save()
            report['gpu_checks']['measured_validation']=run_gpu_stage('measured_validation',args,out,identity,image,allocation,owner,check_cancelled)
            report['gpu_mechanisms_verified']=all(v['gpu_mechanisms_verified'] for v in report['gpu_checks'].values());save()
            for variant in GROUPS:
                check_cancelled();report['stage']='serving_'+variant;save()
                try:
                    report['groups'][variant]=run_group(args,variant,out,'native-ep-'+name+'-'+variant,
                        identity,workload,image,env,owner,client_owner,check_cancelled)
                finally:
                    path=out/variant/'group.json'
                    if path.is_file():report['groups'][variant]=json.loads(path.read_text())
                    save()
            if load_gpqa_workload(args.records,model_path=args.model_path).identity!=workload.identity:
                raise ValueError('data_changed_during_run')
            if profile_summary(load_profiles(args.table_dir,identity))!=identity['tail_profiles']:
                raise ValueError('tail_tables_changed_during_run')
            atomic_json(out/'comparison.json',compare_tail_groups(report['groups']))
            report.update(status='comparison_complete',stage='complete',full_model_graph_execution_verified=True,
                          throughput_result_observed=True);code=0
        if digest(serving_source_manifest())!=source_digest or native_tail_source_sha256()!=identity['native_tail_source_sha256']:
            raise ValueError('source_changed_during_run')
        if checkpoint_stats(args.model_path)!=proof['verified_stats']:
            raise ValueError('checkpoint_changed_during_run')
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__);code=1
        (out/'controller-error-private.log').write_text(str(exc)+'\n')
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={label:pool.submit(child.stop) for label,child in (('server_or_gpu',owner),('client',client_owner))}
            for label,future in futures.items():
                try:future.result()
                except BaseException as exc:
                    report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
                    (out/(label+'-cleanup-error-private.log')).write_text(str(exc)+'\n')
        for variant in ('calibration',*report['groups']):
            if (out/variant).is_dir():copy_service_evidence(args.work_root/('native-ep-'+name+'-'+variant),out/variant)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        try:export_public(out,destination)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(dict(status=report['status'],work_directory=str(out),export_directory=str(destination))),flush=True)
    return code


if __name__=='__main__':
    sys.exit(main())
