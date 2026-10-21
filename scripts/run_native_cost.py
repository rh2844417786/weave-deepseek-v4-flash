#!/usr/bin/env python3
"""Calibrate native MoE costs, validate GPU execution, then compare two 1200s services."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
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
from scripts.check_sglang_service import wait_ready
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

GROUPS = ('native', 'cost')
METHOD = 'native_marlin_measured_cta_v2'
POLICY_VERSION = 'measured_moe_surface_v2'
CANDIDATES = [[8,8],[8,4],[8,2],[4,8],[4,4],[4,2],[2,8],[2,4],[2,2]]


def safe_cost_public(value):
    _safe(value)
    if isinstance(value, dict):
        if {'weights', 'routes', 'raw_routes', 'raw_inputs', 'sorted_token_ids',
            'expert_ids', 'topk_ids', 'topk_weights'}.intersection(value):
            raise ValueError('private_cost_calibration_payload_cannot_be_exported')
        if 'activations' in value and not isinstance(value['activations'], str):
            raise ValueError('private_cost_calibration_payload_cannot_be_exported')
        for child in value.values(): safe_cost_public(child)
    elif isinstance(value, list):
        for child in value: safe_cost_public(child)


def service_command(variant, args, run_name, checkpoint):
    if variant not in (*GROUPS, 'calibration'):
        raise ValueError('unknown native cost arm')
    command = [sys.executable, str(ROOT / 'scripts/run_native_ep.py'), '--execute',
        '--backend', 'native', '--decode-graph', 'full', '--graph-c32', '--request-timing',
        '--work-root', str(args.work_root), '--model-path', str(args.model_path),
        '--run-name', run_name, '--port', str(args.port),
        '--evidence-port-base', str(args.evidence_port_base),
        '--graph-evidence-port-base', str(args.graph_evidence_port_base),
        '--checkpoint-content-sha256', checkpoint]
    if variant in ('cost', 'calibration'):
        flag = '--native-cost-calibrate-dir' if variant == 'calibration' else '--native-cost-table-dir'
        command += ['--native-budget', 'cost', flag, str(args.table_dir),
                    '--native-cost-repeats', str(args.native_cost_repeats)]
        if variant == 'calibration':
            command += ['--native-cost-calibration-timeout', str(math.ceil(args.calibration_timeout))]
    return command


def validate_cost_delta(before, after, graph_check, identity):
    """Bind each GEMM and exactly one planner per MoE to actual graph replays."""
    expected_keys = {(layer, role) for layer in range(43) for role in ('gate_up', 'down')}
    policy = dict(version=POLICY_VERSION, candidate_pairs=CANDIDATES)
    def index(snapshots):
        if len(snapshots) != 4 or {row.get('rank') for row in snapshots} != {0, 1, 2, 3}:
            raise ValueError('four distinct native cost ranks required')
        result = {}
        for snapshot in snapshots:
            rank = snapshot['rank']; profile = identity['cost_profiles'][str(rank)]
            if (snapshot.get('schema_version') != 1 or snapshot.get('method') != METHOD
                    or snapshot.get('mode') != 'cost' or snapshot.get('world_size') != 4
                    or snapshot.get('calibration_complete') is not True
                    or snapshot.get('numerical_policy') != 'record_only' or snapshot.get('policy') != policy
                    or snapshot.get('layer_count') != 43
                    or snapshot.get('source_sha256') != identity['native_cost_source_sha256']
                    or snapshot.get('upstream_sha256') != identity['native_cost_upstream_sha256']
                    or snapshot.get('profile_sha256') != profile['profile_sha256']
                    or snapshot.get('identity') != profile['identity']):
                raise ValueError('native cost source/policy/table/runtime identity differs')
            rows = snapshot.get('rows', [])
            if len(rows) != 86 or {(row.get('layer'), row.get('role')) for row in rows} != expected_keys:
                raise ValueError('complete unique 43-layer two-role cost coverage required')
            for row in rows:
                counters = [row.get(key) for key in ('calls', 'empty_calls', 'native_grid', 'last_active_workers',
                    'last_global_blocks', 'last_local_blocks', 'last_n_tiles', 'last_k_tiles',
                    'last_local_routes', 'last_total_routes')]
                buckets = row.get('bucket_counts', [])
                if (len(buckets) != 9 or any(type(x) is not int or x < 0 for x in counters + buckets)
                        or sum(buckets) != row['calls'] or buckets[0] != row['empty_calls']
                        or row['last_active_workers'] > row['native_grid']
                        or row['last_local_blocks'] > row['last_global_blocks']
                        or row['last_local_routes'] > row['last_total_routes']):
                    raise ValueError('malformed native cost GEMM counters')
            plans = snapshot.get('plans', [])
            if len(plans) != 43 or {p.get('layer') for p in plans} != set(range(43)):
                raise ValueError('complete unique 43-layer planner coverage required')
            for plan in plans:
                v = plan.get('values', [])
                if (len(v) != 32 or any(type(x) is not int or x < 0 for x in v)
                        or v[0] not in (2,4,8) or v[1] not in (2,4,8) or v[20] != 0
                        or v[21] >= 9 or CANDIDATES[v[21]] != v[:2]
                        or sum(v[11:20]) != v[10] or (v[10] and v[9] <= 0)):
                    raise ValueError('malformed native cost plan/counters')
            result[rank] = ({(row['layer'], row['role']): row for row in rows},
                            {p['layer']: p['values'] for p in plans})
        return result
    old, new = index(before), index(after)
    replays = graph_check['replays']
    if type(replays) is not int or replays <= 0:
        raise ValueError('positive actual graph replay count required')
    result = dict(schema_version=1, status='passed', method=METHOD, numerical_policy='record_only',
                  policy=policy, replays=replays, total_calls=0, total_empty_calls=0,
                  total_planner_calls=0, plans_with_pair_variation=0, ranks={})
    for rank in range(4):
        rows, plans = [], []
        for key in sorted(expected_keys):
            a,b = old[rank][0][key],new[rank][0][key]
            calls,empty = b['calls']-a['calls'],b['empty_calls']-a['empty_calls']
            buckets = [end-start for start,end in zip(a['bucket_counts'],b['bucket_counts'])]
            if calls != replays or not 0 <= empty <= calls or any(x < 0 for x in buckets) or sum(buckets) != calls or buckets[0] != empty:
                raise ValueError('native cost GEMM counters do not match actual graph replays')
            rows.append(dict(layer=key[0],role=key[1],calls=calls,empty_calls=empty,bucket_counts=buckets,
                             last_geometry={k:b[k] for k in ('native_grid','last_active_workers',
                                'last_global_blocks','last_local_blocks','last_n_tiles','last_k_tiles',
                                'last_local_routes','last_total_routes')}))
            result['total_calls'] += calls; result['total_empty_calls'] += empty
        for layer in range(43):
            a,b = old[rank][1][layer],new[rank][1][layer]
            calls = b[10]-a[10]; hist = [end-start for start,end in zip(a[11:20],b[11:20])]
            if calls != replays or min(hist) < 0 or sum(hist) != calls:
                raise ValueError('native cost planner must execute once per MoE graph replay')
            observed = [CANDIDATES[i] for i,n in enumerate(hist) if n]
            plans.append(dict(layer=layer,calls=calls,pair_histogram=hist,observed_pairs=observed,
                              last_predicted_moe_ns=b[9],last_plan=b))
            result['total_planner_calls'] += calls
            result['plans_with_pair_variation'] += int(len(observed)>1)
        result['ranks'][str(rank)] = dict(rows=rows,plans=plans)
    result['pair_variation_observed'] = result['plans_with_pair_variation'] > 0
    result['interpretation'] = ('Multiple candidate pairs observed; routing and batch/shape can both affect choices.'
        if result['pair_variation_observed'] else 'No within-layer candidate-pair variation observed in this window.')
    return result


def compare_cost_groups(groups):
    comparison = compare_groups({'native': groups['native'], 'hopper_prefill': groups['cost']})
    rates = comparison['decode_tokens_per_second']
    comparison['cost_vs_native_whole_window_ratio'] = rates['hopper_prefill'] / rates['native']
    comparison['cost_vs_native_sustained_c32_ratio'] = comparison.pop('prefill_vs_native_ratio')
    rates['cost'] = rates.pop('hopper_prefill')
    comparison.update(method=METHOD, complete_weave_claim=False, initial_concurrency=32,
        group_labels={'native': 'Original native SGLang Marlin', 'cost': 'Native Marlin with measured MoE cost budgets'},
        generation_agreement_evaluated=False, prefill_implementation='original_native',
        measurement_mode='native_cost_performance', decode_graph='full')
    return comparison


def export_public(run, destination):
    """Export only aggregate JSON evidence; source data, private logs and overlays stay local."""
    run, destination = Path(run), Path(destination)
    if destination.exists():
        raise FileExistsError('fresh_public_destination_required')
    names = ['experiment.json', 'comparison.json', 'source-manifest.json', 'serving-source-manifest.json',
             'data-manifest.json', 'native-checkpoint-proof.json', 'calibration/check.json', 'calibration/budget-ready.json',
             'calibration/launch.json', 'calibration/runtime.json', 'calibration/server-lifecycle.json',
             'gpu-validation/container-lifecycle.json', 'gpu-validation/check.json']
    names += [f'calibration/native-cost-rank{rank}.json' for rank in range(4)]
    names += [f'gpu-validation/native-cost-gpu-rank{rank}.json' for rank in range(4)]
    group_names = ('group.json', 'launch.json', 'runtime.json', 'server-lifecycle.json',
        'measurement-lifecycle.json', 'client-summary.json', 'measurement-summary.json',
        'gpqa-rows.json', 'load-window.json', 'load-progress.json', 'warmup-ready.json',
        'scheduler-summary.json', 'graph-before.json', 'graph-after.json', 'graph-check.json',
        'budget-before.json', 'budget-after.json', 'budget-check.json')
    names += [f'{group}/{name}' for group in GROUPS for name in group_names]
    files = {}
    for name in names:
        path = run / name
        if path.is_symlink():
            raise ValueError('public_symlink_forbidden')
        if path.is_file():
            raw = path.read_bytes(); safe_cost_public(json.loads(raw)); files[name] = raw
    destination.mkdir(parents=True)
    for name, raw in files.items():
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw); path.chmod(0o664)
    manifest = dict(schema_version=1, status=json.loads(files['experiment.json'])['status'],
                    raw_content_exported=False, files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    atomic_json(destination / 'manifest.json', manifest); (destination / 'manifest.json').chmod(0o664)
    return manifest


def copy_service_evidence(server_dir, folder):
    for source, target in (('launch.json', 'launch.json'), ('runtime.json', 'runtime.json'),
                           ('container_lifecycle.json', 'server-lifecycle.json')):
        if (server_dir / source).is_file():
            atomic_json(folder / target, json.loads((server_dir / source).read_text()))


def run_group(args, variant, out, run_name, identity, workload, image, env, owner, client_owner, check_cancelled):
    folder = out / variant; folder.mkdir()
    server_dir = args.work_root / run_name; url = f'http://127.0.0.1:{args.port}'
    group = dict(status='running', variant=variant, stage='starting', identity=identity)
    ports = [args.port] + list(range(args.evidence_port_base, args.evidence_port_base + 4)) + list(range(args.graph_evidence_port_base, args.graph_evidence_port_base + 4))
    try:
        assert_ports_free(ports)
        with (folder / 'server-launch-private.log').open('w') as log:
            server = owner.spawn(service_command(variant, args, run_name, identity['checkpoint_content_sha256']),
                server_dir / 'container_lifecycle.json', env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(url, timeout=args.setup_timeout, process=server)
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
                    if variant == 'cost':
                        budget_before = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='budget-coverage')
                        atomic_json(folder / 'budget-before.json', budget_before)
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
            if variant == 'cost':
                budget_after = baseline_coverage(args.graph_evidence_port_base, deadline=idle_deadline, endpoint='budget-coverage')
                atomic_json(folder / 'budget-after.json', budget_after)
                budget_check = validate_cost_delta(budget_before, budget_after, graph_check, identity)
                atomic_json(folder / 'budget-check.json', budget_check); group['budget_evidence'] = budget_check
            group.update(status='passed', stage='complete', client_summary=client, batch_evidence=graph_check)
        owner.stop(); check_cancelled()
    except BaseException as exc:
        group.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(folder / 'group.json', group)
        copy_service_evidence(server_dir, folder)
    return group


def load_profiles(directory, identity):
    """Core schema owns table geometry; the controller binds the current experiment."""
    from src.weave.native_cost import load_profile
    expected = dict(cost_source_sha256=identity['native_cost_source_sha256'],
                    upstream_sha256=identity['native_cost_upstream_sha256'],
                    image_id=identity['image_id'],
                    checkpoint_content_sha256=identity['checkpoint_content_sha256'])
    profiles = {}
    for rank in range(4):
        path = Path(directory) / f'native-cost-rank{rank}.json'
        if path.is_symlink() or not path.is_file():
            raise ValueError('complete native cost tables required; generate with python3 '
                             'scripts/run_native_cost.py --work-root WORK_ROOT --records PREPARED_MAIN --execute')
        profile = load_profile(path, expected_identity=dict(expected, rank=rank))
        safe_cost_public(profile)
        for batches in profile['tables'].values():
            for table in batches.values():
                validation = table['validation']
                if (validation.get('complete') is not True
                        or validation.get('numerical_policy') != 'record_only'
                        or len(validation.get('heldout', [])) != 2):
                    raise ValueError('two independent complete held-out routes required per class/batch')
        profiles[str(rank)] = profile
    # GPU identity can differ by rank; the model, source and graph contracts cannot.
    first = profiles['0']
    for profile in profiles.values():
        if profile['groups'] != first['groups']:
            raise ValueError('structural class mapping differs across ranks')
        for key in ('runtime', 'precision', 'model_geometry', 'model_config_sha256', 'sglang_commit', 'graph_contract'):
            if profile['identity'][key] != first['identity'][key]:
                raise ValueError('calibration identity differs across ranks: ' + key)
    return profiles


def profile_summary(profiles):
    return {rank: {'identity': profile['identity'], 'profile_sha256': profile['profile_sha256'],
                   'groups': profile['groups']} for rank, profile in profiles.items()}


def calibration_ready(snapshots, profiles):
    if len(snapshots) != 4 or {s.get('rank') for s in snapshots} != set(range(4)):
        raise ValueError('four calibration completion snapshots required')
    for snapshot in snapshots:
        profile = profiles[str(snapshot['rank'])]
        if (snapshot.get('method') != METHOD or snapshot.get('mode') != 'calibration'
                or snapshot.get('world_size') != 4 or snapshot.get('layer_count') != 43
                or snapshot.get('calibration_complete') is not True
                or snapshot.get('numerical_policy') != 'record_only'
                or snapshot.get('profile_sha256') != profile['profile_sha256']
                or snapshot.get('identity') != profile['identity']):
            raise ValueError('calibration ready proof differs from complete rank profiles')


def run_calibration(args, out, run_name, identity, env, owner, check_cancelled):
    folder = out / 'calibration'; folder.mkdir(exist_ok=True)
    server_dir = args.work_root / run_name
    result = dict(status='running', mode='calibration', numerical_policy='record_only')
    try:
        assert_ports_free([args.port] + list(range(args.graph_evidence_port_base, args.graph_evidence_port_base + 4)))
        with (folder / 'server-launch-private.log').open('w') as log:
            server = owner.spawn(service_command('calibration', args, run_name, identity['checkpoint_content_sha256']),
                server_dir / 'container_lifecycle.json', env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(f'http://127.0.0.1:{args.port}', timeout=args.calibration_timeout, process=server)
            check_cancelled()
            validate_runtime(json.loads((server_dir / 'runtime.json').read_text()), identity, env['CUDA_VISIBLE_DEVICES'])
            profiles = load_profiles(args.table_dir, identity)
            snapshots = baseline_coverage(args.graph_evidence_port_base, deadline=time.monotonic()+60., endpoint='budget-coverage')
            calibration_ready(snapshots, profiles)
            atomic_json(folder / 'budget-ready.json', snapshots)
            result.update(status='complete', service_ready=True, profiles=profile_summary(profiles),
                          heldout_validation='complete_record_only', calibration_complete=True)
        return profiles
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        try: owner.stop()
        except BaseException as exc:
            result.update(status='failed', cleanup_error_type=type(exc).__name__)
            raise
        finally:
            copy_service_evidence(server_dir, folder)
            atomic_json(folder / 'check.json', result)


def gpu_command(args, image, out, allocation=None, checkpoint=None):
    """Pinned existing image and allocated devices; placeholders are preview-only."""
    relative = out.relative_to(args.work_root)
    table_relative = args.table_dir.relative_to(args.work_root)
    name = 'weave-' + out.parent.name + '-gpu'
    command = ['docker', 'run', '--rm', '--pull=never', '--name', name,
        '--gpus', allocation.docker_argument if allocation else '<GPUQ allocation>',
        '--network=host', '--shm-size=8g', '-v', f'{ROOT}:/workspace:ro',
        '-v', f'{args.work_root}:/work', '-v', f'{args.model_path}:/model:ro', '-w', '/workspace',
        '-e', 'CUDA_VISIBLE_DEVICES=' + (allocation.container_visible if allocation else '<container logical devices>'),
        '-e', 'WEAVE_HOST_ALLOCATION=' + (allocation.csv if allocation else '<GPUQ allocation>'),
        '-e', f'WEAVE_IMAGE_ID={image}', '-e', f'WEAVE_CHECKPOINT_CONTENT_SHA256={checkpoint or "0"*64}',
        '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'NCCL_SOCKET_IFNAME=lo', '-e', 'NCCL_IB_DISABLE=1',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_DATASETS_OFFLINE=1',
        '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
        image, load_contract()['python_executable'], '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=4', '/workspace/tests/gpu/check_native_cost.py',
        '--model-path', '/model', '--output-dir', f'/work/{relative}',
        '--cache-root', '/work/cache/weave-native-cost', '--table-dir', f'/work/{table_relative}',
        '--timeout-seconds', str(args.gpu_timeout)]
    return command


def run_gpu_validation(args, out, identity, image, allocation, owner, check_cancelled):
    folder = out / 'gpu-validation'; folder.mkdir()
    command = gpu_command(args, image, folder, allocation, identity['checkpoint_content_sha256'])
    lifecycle = folder / 'container-lifecycle.json'
    manager = [sys.executable, str(ROOT / 'scripts/manage_container.py'), '--name', command[command.index('--name')+1],
               '--evidence', str(lifecycle), '--log', str(folder / 'gpu-private.log'), '--'] + command
    result = dict(status='running', method=METHOD, numerical_policy='record_only')
    try:
        child = owner.spawn(manager, lifecycle, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.gpu_timeout + 120.
        while child.poll() is None:
            check_cancelled()
            if time.monotonic() >= deadline: raise TimeoutError('native_cost_gpu_validation_deadline')
            time.sleep(.2)
        if child.returncode != 0: raise RuntimeError('native_cost_gpu_validation_failed')
        owner.stop(); check_cancelled()
        reports = {}
        for rank in range(4):
            report = json.loads((folder / f'native-cost-gpu-rank{rank}.json').read_text())
            safe_cost_public(report)
            if (report.get('rank') != rank or report.get('status') != 'complete'
                    or report.get('checks_complete') is not True
                    or report.get('cleanup', {}).get('status') != 'complete'
                    or type(report.get('case_count')) is not int or report['case_count'] <= 0
                    or report.get('case_count') != report.get('expected_case_count')
                    or report.get('profile_sha256') != identity['cost_profiles'][str(rank)]['profile_sha256']):
                raise ValueError('complete native cost GPU coverage, profile binding and cleanup required')
            reports[str(rank)] = {key: report[key] for key in ('status','case_count','expected_case_count','profile_sha256','cleanup')}
        result.update(status='complete', ranks=reports)
        return result
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(folder / 'check.json', result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-root', 'records'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--native-baseline-dir', type=Path, default=ROOT / 'reports/server/gpqa-c32-9f1635df2afe')
    parser.add_argument('--calibration-dir', type=Path, help='explicit reuse of complete matching rank profiles under WORK_ROOT')
    parser.add_argument('--native-cost-repeats', type=int, default=7)
    parser.add_argument('--export-root', type=Path)
    parser.add_argument('--calibration-timeout', type=float, default=43200.)
    parser.add_argument('--gpu-timeout', type=int, default=7200)
    parser.add_argument('--setup-timeout', type=float, default=7200.)
    parser.add_argument('--warmup-timeout', type=float, default=960.)
    parser.add_argument('--drain-timeout', type=float, default=60.)
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--evidence-port-base', type=int, default=30110)
    parser.add_argument('--graph-evidence-port-base', type=int, default=30310)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.native_cost_repeats < 2: parser.error('native cost requires at least 2 repeats')
    for key in ('work_root', 'records', 'model_path', 'native_baseline_dir', 'export_root', 'calibration_dir'):
        path = getattr(args, key)
        if path is not None:
            if not path.is_absolute() or '..' in path.parts: parser.error(key + ' must be absolute without traversal')
            setattr(args, key, path.resolve())
    if args.calibration_dir:
        try: args.calibration_dir.relative_to(args.work_root)
        except ValueError: parser.error('calibration-dir must be under WORK_ROOT')
    if any(not math.isfinite(x) or x <= 0 for x in (args.calibration_timeout, args.setup_timeout, args.warmup_timeout, args.drain_timeout)):
        parser.error('finite positive timeouts required')
    if not 30 <= args.gpu_timeout <= 7200: parser.error('GPU timeout must be in 30..7200 seconds')
    ports = [args.port] + list(range(args.evidence_port_base,args.evidence_port_base+4)) + list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4))
    if min(ports)<1 or max(ports)>65535 or len(set(ports))!=len(ports): parser.error('valid distinct serving/evidence ports required')
    name = 'native-cost-' + uuid.uuid4().hex[:12]
    out = args.work_root / name; destination = (args.export_root or args.work_root / 'public-reports') / name
    args.table_dir = args.calibration_dir or out / 'calibration'
    contract = load_contract()
    if not args.execute:
        print(json.dumps(dict(status='preview_not_executed',method=METHOD,numerical_policy='record_only',
            stages=['reuse_calibration' if args.calibration_dir else 'calibration','gpu_validation',*GROUPS],
            calibration=service_command('calibration',args,'native-ep-'+name+'-calibration','0'*64) if not args.calibration_dir else None,
            gpu_validation=gpu_command(args,contract['image'],out/'gpu-validation'),
            groups={variant:service_command(variant,args,'native-ep-'+name+'-'+variant,'0'*64) for variant in GROUPS},
            clients={variant:client_command(contract['image'],args.work_root,args.records,f'{name}/{variant}',name+'-'+variant,args.port) for variant in GROUPS},
            seconds_per_arm=SECONDS,initial_concurrency=32,prefill_implementation='original_native',decode_graph='full',
            complete_weave_claim=False,calibration_repeats=args.native_cost_repeats,calibration_directory=str(args.table_dir),
            required_environment={'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'},
            evidence_reads='idle_before_and_after_only',native_baseline_dir=str(args.native_baseline_dir)),indent=2))
        return 0
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False)
    owner,client_owner=OwnedChild(),OwnedChild()
    report=dict(schema_version=1,status='running',stage='validating',method=METHOD,
        numerical_policy='record_only',generation_comparison_policy='record_only',generation_agreement_evaluated=False,
        complete_weave_claim=False,initial_concurrency=32,seconds_per_arm=SECONDS,groups={},
        started_utc=datetime.now(timezone.utc).isoformat(),gpuq_job_id=os.environ.get('GPUQ_JOB_ID'),host_gpu_allocation=allocation.csv)
    code=1
    def save(): atomic_json(out/'experiment.json',report)
    def cancel(number,frame):
        for child in (owner,client_owner): child.requested=number
        if not any(child.protected for child in (owner,client_owner)): raise InterruptedError('job_cancelled')
    def check_cancelled(): owner.check_cancelled();client_owner.check_cancelled()
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        save();workload=load_gpqa_workload(args.records,model_path=args.model_path)
        source=serving_source_manifest();source_digest=digest(source)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        if not image.startswith('sha256:'): raise ValueError('content-addressed local image required')
        proof=native_checkpoint_proof(args.native_baseline_dir,args.model_path)
        from src.weave.native_cost import native_cost_source_sha256
        from src.weave.graph_hooks import graph_observer_source_sha256
        identity=dict(source_sha256=digest(kernel_source_manifest()),checkpoint_content_sha256=proof['checkpoint_content_sha256'],
            image_id=image,serving_source_sha256=source_digest,decode_graph='full',**workload.identity,
            native_cost_source_sha256=native_cost_source_sha256(),
            native_cost_upstream_sha256=json.loads((ROOT/'configs/native_budget_sources.json').read_text())['sha256'],
            observer_source_sha256=graph_observer_source_sha256(),graph_sources=json.loads((ROOT/'configs/sglang_graph_sources.json').read_text())['sha256'])
        report['identity']=identity;save()
        for file,value in (('native-checkpoint-proof.json',proof),('data-manifest.json',workload.identity),
                           ('source-manifest.json',kernel_source_manifest()),('serving-source-manifest.json',source)):
            atomic_json(out/file,value)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
             'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        check_cancelled();report.update(stage='calibration');save()
        if args.calibration_dir:
            profiles=load_profiles(args.table_dir,identity)
            folder=out/'calibration';folder.mkdir()
            for rank,profile in profiles.items(): atomic_json(folder/f'native-cost-rank{rank}.json',profile)
            atomic_json(folder/'check.json',dict(status='complete',mode='explicit_profile_reuse',
                profiles=profile_summary(profiles),heldout_validation='complete_record_only',numerical_policy='record_only'))
        else:
            profiles=run_calibration(args,out,'native-ep-'+name+'-calibration',identity,env,owner,check_cancelled)
        identity['cost_profiles']=profile_summary(profiles);save()
        check_cancelled();report.update(stage='gpu_validation');save()
        report['gpu_validation']=run_gpu_validation(args,out,identity,image,allocation,owner,check_cancelled);save()
        for variant in GROUPS:
            check_cancelled();report.update(stage='serving_'+variant);save()
            try:
                report['groups'][variant]=run_group(args,variant,out,'native-ep-'+name+'-'+variant,
                    identity,workload,image,env,owner,client_owner,check_cancelled)
            finally:
                path=out/variant/'group.json'
                if path.is_file(): report['groups'][variant]=json.loads(path.read_text())
                save()
        if load_gpqa_workload(args.records,model_path=args.model_path).identity!=workload.identity: raise ValueError('data_changed_during_run')
        if digest(serving_source_manifest())!=source_digest or native_cost_source_sha256()!=identity['native_cost_source_sha256']: raise ValueError('source_changed_during_run')
        if checkpoint_stats(args.model_path)!=proof['verified_stats']: raise ValueError('checkpoint_changed_during_run')
        if profile_summary(load_profiles(args.table_dir,identity))!=identity['cost_profiles']: raise ValueError('cost_tables_changed_during_run')
        atomic_json(out/'comparison.json',compare_cost_groups(report['groups']))
        report.update(status='comparison_complete',stage='complete');code=0
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        (out/'controller-error-private.log').write_text(str(exc)+'\n')
    finally:
        for sig in previous: signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={label:pool.submit(child.stop) for label,child in (('server_or_gpu',owner),('client',client_owner))}
            for label,future in futures.items():
                try: future.result()
                except BaseException as exc:
                    report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
                    (out/(label+'-cleanup-error-private.log')).write_text(str(exc)+'\n')
        for variant in ('calibration',*report['groups']):
            if (out/variant).is_dir(): copy_service_evidence(args.work_root/('native-ep-'+name+'-'+variant),out/variant)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        try: export_public(out,destination)
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
    print(json.dumps(dict(status=report['status'],work_directory=str(out),export_directory=str(destination))),flush=True)
    return code


if __name__=='__main__': sys.exit(main())
