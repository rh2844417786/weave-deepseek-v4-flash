"""Four-H100 FP8/BF16 local expert compute gate over all private captured layers.

torchrun --nproc-per-node=4 tests/gpu/check_hopper_candidate.py --model-path /model
    --capture-dir /capture --output-dir /work/fresh --identity-file /work/identity.json
    --invocation-id UUID --warmups 3 --repetitions 20 --budget-seconds 7200

Only Gloo coordinates stage failures and budget stops. Timed functions execute
rank-local online grouping through routed FP32 slots, without IPC or NCCL.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import (SCOPE, SYNTHETIC_CASES, VARIANTS, atomic_report,
    public_resources, replay_source_sha256, validate_identity, validate_rank_report)
from src.weave.replay_trace import validate_capture_id


class AgreedFailure(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require_fresh_output(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    # The parent/preflight already writes identity, manifests and runtime.json.
    # Refuse previous worker results, not the metadata needed to start this job.
    if (any(path.glob('hopper-rank*.json')) or any(path.glob('hopper-progress-rank*.json'))
            or (path/'summary.json').exists()):
        raise ValueError('fresh worker output required; old timing points cannot be reused')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'capture-dir', 'output-dir', 'identity-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--invocation-id', required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--budget-seconds', type=float, default=7200)
    args = parser.parse_args(argv)
    if args.warmups < 1 or args.repetitions < 1 or not 0 < args.budget_seconds <= 86400:
        parser.error('positive warmups/repetitions and budget up to 86400 seconds required')
    try:
        validate_capture_id(args.invocation_id)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def agreed(label, operation):
    """Every rank executes exactly one agreement after each local operation.

    Never transmit exception text: assertions from tensors can include private
    values. Detailed source locations remain available by the fixed stage name.
    """
    result, failure = None, None
    try:
        result = operation()
    except Exception as exc:
        failure = type(exc).__name__
    failures = [None] * 4
    dist.all_gather_object(failures, failure)
    if any(failures):
        print(json.dumps({'stage': label, 'rank_failure_types': failures}), flush=True)
        raise AgreedFailure(label)
    return result


def numerical(actual, expected):
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    if not finite:
        return {'finite': False, 'relative_l2': None, 'max_abs': None}
    difference = actual.double() - expected.double()
    relative = (difference.norm() / expected.double().norm().clamp_min(1e-12)).item()
    maximum = difference.abs().max().item() if difference.numel() else 0.
    if not math.isfinite(relative) or not math.isfinite(maximum):
        return {'finite': False, 'relative_l2': None, 'max_abs': None}
    return {'finite': True, 'relative_l2': relative, 'max_abs': maximum}


def numerical_failure(comparisons):
    """Only nonfinite execution is fatal; all finite differences are recorded."""
    return 'nonfinite_output' if any(not metric['finite'] for checks in comparisons
                                   for metric in checks.values()) else None


def local_reference(x, ids, routing, raw, rank, experts, clamp, oracle):
    local = experts // 4
    products = torch.zeros((x.shape[0] * ids.shape[1], x.shape[1]), device=x.device, dtype=torch.float32)
    for expert in sorted(set(ids.cpu().flatten().tolist())):
        if expert // local == rank:
            token, slot = torch.where(ids == expert)
            down = oracle(x[token], raw, expert % local, clamp)
            products[token * ids.shape[1] + slot] = (
                down.bfloat16() * routing[token, slot].bfloat16().unsqueeze(1)).float()
    return products


def workspace_pointers(baseline, candidate):
    return tuple(getattr(ws, name).data_ptr() for ws, names in (
        (baseline, ('online_slot_outputs', 'online_group_meta', 'online_group_slots', 'online_group_count')),
        (candidate, ('slot_outputs', 'group_counts', 'active_experts', 'active_count', 'expert_slots')))
        for name in names)


def metadata(candidate, baseline, ids, rank, experts):
    rows = ids.cpu().tolist()
    local = experts // 4
    expected = [[] for _ in range(local)]
    for token, selected in enumerate(rows):
        for slot, expert in enumerate(selected):
            if expert // local == rank:
                expected[expert % local].append(token * len(selected) + slot)
    counts = [len(slots) for slots in expected]
    active = [expert for expert, count in enumerate(counts) if count]
    slots = [items + [-1] * (32 - len(items)) for items in expected]
    if (candidate.group_counts.cpu().tolist() != counts
            or candidate.active_count.cpu().tolist() != [len(active)]
            or candidate.active_experts.cpu().tolist() != active + [-1] * (local - len(active))
            or candidate.expert_slots.cpu().tolist() != slots):
        raise AssertionError('candidate grouping metadata differs from ordered input slots')
    plan = build_staged_groups(rows, rank=rank, ep=4, experts=experts)
    if baseline.online_group_count.cpu().tolist() != [plan.group_count]:
        raise AssertionError('BF16 group count differs from CPU grouping')
    return {'batch': len(rows), 'local_slots': sum(counts), 'active_experts': len(active),
            'max_rows_per_expert': max(counts), 'bf16_groups': plan.group_count}


def evaluate(baseline, candidate, calls, inputs, raw, rank, experts, clamp, expected=None):
    x, ids, routing = inputs
    if expected is None:
        expected = (
            local_reference(x, ids, routing, raw, rank, experts, clamp, expert_reference),
            local_reference(x, ids, routing, raw, rank, experts, clamp, expert_reference_fp8))
    baseline.online_slot_outputs.fill_(float('nan'))
    candidate.slot_outputs.fill_(float('nan'))
    calls[0](); calls[1]()
    torch.cuda.synchronize()
    rows = x.shape[0] * ids.shape[1]
    bf16 = baseline.online_slot_outputs[:rows]
    fp8 = candidate.slot_outputs[:rows]
    checks = {'baseline': numerical(bf16, expected[0]),
              'implementation': numerical(fp8, expected[1]),
              'precision': numerical(fp8, bf16)}
    workload = metadata(candidate, baseline, ids, rank, experts)
    remote = (ids // (experts // 4) != rank).flatten()
    nonlocal_zero = not (torch.count_nonzero(bf16[remote]).item() or torch.count_nonzero(fp8[remote]).item())
    if not nonlocal_zero:
        raise AssertionError('nonlocal slots must be overwritten with zero')
    # Shrinking B may not leave stale values in the candidate's fixed-capacity output.
    if torch.count_nonzero(candidate.slot_outputs[rows:]).item():
        raise AssertionError('unused candidate output slots must be zero')
    return checks, workload, expected


def graphs(calls):
    """Warm up on a side stream, then capture both identical compute scopes."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            calls[0](); calls[1]()
    stream.synchronize()
    captured = []
    for call in calls:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        captured.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return captured


def event_samples(operation, warmups, repetitions):
    # Event allocation/first-record initialization and all correctness checks are
    # excluded. The sole interval contains online grouping through local slots.
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(repetitions)]
    for start, end in pairs:
        start.record(); end.record()
    torch.cuda.synchronize()
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    values = []
    for start, end in pairs:
        start.record(); operation(); end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    if any(not 0 < value < float('inf') for value in values):
        raise AssertionError('event timing must be finite and positive')
    return values


def synthetic_weights(rank, device):
    shapes = [(8, 128, 64)] * 3 + [(8, 128, 4)] * 3
    raw = []
    for number, shape in enumerate(shapes):
        count = 1
        for size in shape:
            count *= size
        values = torch.arange(count, device=device, dtype=torch.int32)
        # Every signed E2M1 nibble and finite block-scale pattern appears. Raw
        # identities vary by rank; scales remain modest enough for BF16 tests.
        values = ((values + rank * 17 + number * 13) % 256 if number < 3
                  else 119 + (values + rank + number) % 6)
        raw.append(values.to(torch.uint8).reshape(shape))
    return raw


def synthetic_inputs(batch, variant, device, *, zero=False):
    generator = torch.Generator().manual_seed(71237 + variant)
    x = (torch.randn((batch, 128), generator=generator) * .125).bfloat16().to(device)
    if zero:
        x.zero_()
    if variant % 3:
        rows = [[((variant + 1) % 4) * 8 + slot for slot in range(6)] for _ in range(batch)]
    else:
        rows = [[(token * 7 + slot * 5 + variant) % 32 for slot in range(6)] for token in range(batch)]
    ids = torch.tensor(rows, device=device, dtype=torch.int64 if variant % 2 else torch.int32)
    weights = [.1, .15, .2, .1, .25, .2] if variant % 2 else [.2, .1, .1, .25, .15, .2]
    routing = torch.tensor([weights] * batch, device=device,
                           dtype=torch.bfloat16 if variant % 2 else torch.float32)
    return x, ids, routing


def synthetic_gate(modules, rank, device, clamp, report, save):
    baseline = modules[0].StagedWorkspace(rank, 4, 32, 128, 128, 32, 6)
    candidate = modules[1].FP8Workspace(rank, 4, 32, 128, 128, 32, 6)
    raw = synthetic_weights(rank, device)
    bound = (baseline.bind_online_weights(raw), candidate.bind_weights(raw))
    before = workspace_pointers(baseline, candidate)
    saw_empty = False
    fixed = None
    captured = None
    for variant, (name, batch, mode) in enumerate(SYNTHETIC_CASES):
        inputs = synthetic_inputs(batch, variant, device, zero=name == 'zero_b32')
        if mode == 'graph':
            if fixed is None:
                # Capture with different contents, then change all three input
                # tensors in place before each oracle check and any timing.
                fixed = synthetic_inputs(32, 123, device)
                eager = (lambda: baseline.run_online(*fixed, bound[0], clamp),
                         lambda: candidate.run(*fixed, bound[1], clamp))
                captured = graphs(eager)
            for destination, source in zip(fixed, inputs):
                destination.copy_(source)
            inputs = fixed
            calls = (captured[0].replay, captured[1].replay)
        else:
            calls = (lambda: baseline.run_online(*inputs, bound[0], clamp),
                     lambda: candidate.run(*inputs, bound[1], clamp))
        checks, workload, _ = evaluate(baseline, candidate, calls, inputs, raw, rank, 32, clamp)
        saw_empty |= workload['local_slots'] == 0
        report['synthetic']['cases'].append({'case': name, 'batch': batch, 'mode': mode,
            'checks': checks, 'metadata_exact': True, 'nonlocal_slots_zero': True})
        if mode == 'graph':
            report['synthetic']['graph_input_updates'] += 1
        if numerical_failure([checks]) is not None:
            report['synthetic']['status'] = 'failed'
            save()
            raise NumericalFailure('nonfinite_output')
    if before != workspace_pointers(baseline, candidate) or not saw_empty:
        raise AssertionError('stable workspace and empty local rank must be exercised')
    report['synthetic'].update(status='passed', stable_addresses=True, fp4_patterns=True, empty_rank=True)
    save()


def main(argv=None):
    args = parse_args(argv)
    global torch, dist, build_staged_groups, expert_reference, expert_reference_fp8
    import torch
    import torch.distributed as dist
    from src.weave.checkpoint_identity import checkpoint_identity, checkpoint_stats, sha256_file
    from src.weave.hopper_candidate import extension as fp8_extension, source_sha256
    from src.weave.hopper_reference import expert_reference_fp8
    from src.weave.ops import extension as bf16_extension, kernel_source_sha256
    from src.weave.replay_trace import (tokenizer_manifest_sha256,
        load_rank_trace, validate_capture_directory)
    from src.weave.staged_plan import build_staged_groups
    from src.weave.weights import load_raw_experts
    from tests.gpu.check_checkpoint import expert_reference

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    if dist.get_world_size() != 4 or dist.get_rank() != rank:
        raise ValueError('single-node torchrun with exactly four allocated H100 ranks required')
    stop = [False]
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda signum, frame: stop.__setitem__(0, True))
    started = None
    report = None

    def progress(stage):
        atomic_report(args.output_dir / f'hopper-progress-rank{rank}.json',
            {'invocation_id': args.invocation_id, 'stage': stage,
             'measurement_started_monotonic': started})

    def save():
        atomic_report(args.output_dir / f'hopper-rank{rank}.json', report)

    def budget():
        flags = [None] * 4
        dist.all_gather_object(flags, (stop[0], started is not None and time.monotonic() - started >= args.budget_seconds))
        if any(flag[0] for flag in flags):
            raise NumericalFailure('interrupted')
        if any(flag[1] for flag in flags):
            raise BudgetStop()

    def fresh_directory():
        if rank == 0:
            require_fresh_output(args.output_dir)
    agreed('fresh_directory', fresh_directory)
    agreed('identity_progress', lambda: progress('identity'))

    try:
        def inspect():
            identity = json.loads(args.identity_file.read_text())
            validate_identity(identity)
            if (identity['invocation_id'] != args.invocation_id or identity['warmups'] != args.warmups
                    or identity['repetitions'] != args.repetitions
                    or identity['candidate_source_sha256'] != source_sha256()
                    or identity['bf16_source_sha256'] != kernel_source_sha256()
                    or identity['replay_source_sha256'] != replay_source_sha256()):
                raise ValueError('invocation/current source identity mismatch')
            current = identity['capture_identity']
            if (current['tokenizer_manifest_sha256'] != tokenizer_manifest_sha256(args.model_path)
                    or identity['capture_manifest_sha256'] != sha256_file(args.capture_dir / 'capture_manifest.json')):
                raise ValueError('capture tokenizer/manifest identity mismatch')
            validate_capture_directory(args.capture_dir, current)
            config = json.loads((args.model_path / 'config.json').read_text())
            if tuple(config[key] for key in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts',
                    'num_experts_per_tok', 'num_hidden_layers')) != (4096, 2048, 256, 6, 43):
                raise ValueError('original 43-layer H4096 I2048 E256 top6 checkpoint required')
            props = torch.cuda.get_device_properties(rank)
            runtime = {'device_name': props.name, 'capability': [props.major, props.minor],
                'multiprocessors': props.multi_processor_count, 'device_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda, 'torch_version': str(torch.__version__)}
            if props.major != 9 or props.minor != 0 or 'H100' not in props.name or runtime['device_count'] != 4:
                raise ValueError('four visible H100 devices required')
            return identity, float(config['swiglu_limit']), runtime, checkpoint_stats(args.model_path)
        identity, clamp, runtime, initial_stats = agreed('inspect_identity', inspect)
        content = agreed('checkpoint_content', lambda: checkpoint_identity(args.model_path) if rank == 0 else None)
        hashes = [content['sha256'] if rank == 0 else None]
        dist.broadcast_object_list(hashes, src=0)
        agreed('checkpoint_identity_match', lambda: require(hashes[0] == identity['checkpoint_content_sha256']))

        def replicas():
            # Explicit CPU tensor equality across ALL four validated payloads.
            # No frozen activations, IDs or routing values enter public reports.
            first = load_rank_trace(args.capture_dir, 0, identity['capture_identity'])
            for other_rank in (1, 2, 3):
                other = load_rank_trace(args.capture_dir, other_rank, identity['capture_identity'])
                if any(not torch.equal(first[name], other[name]) for name in ('x', 'ids', 'routing')):
                    raise ValueError('captured TP replicas are unequal')
                del other
            digest = hashlib.sha256()
            for name in ('x', 'ids', 'routing'):
                digest.update(name.encode() + b'\0')
                digest.update(first[name].view(torch.uint8).numpy().tobytes())
            return first, digest.hexdigest()
        trace, replica_sha = agreed('replica_equality', replicas)
        report = {'schema_version': 2, 'kind': 'hopper_fp8_local_expert_replay', 'rank': rank,
            'status': 'partial', 'identity': identity, 'measurement_scope': SCOPE,
            'used_for_formal_performance': False, 'service_promotion_allowed': False,
            'replica_sha256': replica_sha, 'synthetic': {'status': 'pending', 'cases': [],
                'stable_addresses': False, 'graph_input_updates': 0, 'fp4_patterns': False, 'empty_rank': False},
            'points': [], 'resources': {'bf16': {}, 'fp8': {}}, 'runtime': runtime, 'stop_reason': 'none'}
        agreed('initial_report', save)
        agreed('compile_progress', lambda: progress('compile'))
        modules = agreed('compile_extensions', lambda: (bf16_extension(), fp8_extension()))
        device = torch.device('cuda', rank)
        agreed('synthetic_progress', lambda: progress('synthetic'))
        agreed('synthetic_gate', lambda: synthetic_gate(modules, rank, device, clamp, report, save))
        budget()

        def allocate():
            baseline = modules[0].StagedWorkspace(rank, 4, 32, 4096, 2048, 256, 6)
            candidate = modules[1].FP8Workspace(rank, 4, 32, 4096, 2048, 256, 6)
            fixed = (torch.empty((32, 4096), device=device, dtype=torch.bfloat16),
                     torch.empty((32, 6), device=device, dtype=torch.int32),
                     torch.empty((32, 6), device=device, dtype=torch.float32))
            report['resources'] = {'bf16': public_resources(baseline.resources()), 'fp8': public_resources(candidate.resources())}
            return baseline, candidate, fixed
        baseline, candidate, fixed = agreed('allocate_real_workspace', allocate)
        before = agreed('initial_real_addresses', lambda: workspace_pointers(baseline, candidate))
        started = time.monotonic()
        agreed('measurement_progress', lambda: progress('measurement'))
        for layer in range(43):
            budget()
            raw = agreed('load_layer', lambda: load_raw_experts(args.model_path, layer, rank, 4, device=device))
            bound = agreed('bind_layer', lambda: (baseline.bind_online_weights(raw), candidate.bind_weights(raw)))
            eager = (lambda: baseline.run_online(*fixed, bound[0], clamp),
                     lambda: candidate.run(*fixed, bound[1], clamp))
            # Graphs initially capture sample 128; sample 64 is copied in place
            # before the first graph check. Thus each timed graph has already
            # consumed updated live inputs, including at the first real sample.
            agreed('graph_initial_inputs', lambda: copy_trace(fixed, trace, 1, layer))
            captured = agreed('capture_graphs', lambda: graphs(eager))
            operations = {'bf16_eager': eager[0], 'fp8_eager': eager[1],
                          'bf16_graph': captured[0].replay, 'fp8_graph': captured[1].replay}
            for sample_index, sample in enumerate((64, 128)):
                budget()
                agreed('copy_real_inputs', lambda: copy_trace(fixed, trace, sample_index, layer))
                eager_result = agreed('real_eager_oracles', lambda: evaluate(
                    baseline, candidate, eager, fixed, raw, rank, 256, clamp))
                graph_result = agreed('real_graph_oracles', lambda: evaluate(
                    baseline, candidate, (captured[0].replay, captured[1].replay),
                    fixed, raw, rank, 256, clamp, eager_result[2]))
                agreed('stable_real_addresses', lambda: require(before == workspace_pointers(baseline, candidate)))
                local_failure = numerical_failure([eager_result[0],graph_result[0]])
                failures = [None] * 4
                dist.all_gather_object(failures, local_failure)
                shift = (layer * 2 + sample_index) % 4
                order = VARIANTS[shift:] + VARIANTS[:shift]
                for variant in order:
                    checks, workload, _ = graph_result if variant.endswith('graph') else eager_result
                    point = {'layer': layer, 'sample': sample, 'variant': variant, 'gpu_ms': [],
                        'timing_valid': False, 'checks': checks, 'workload': workload,
                        'stable_addresses': True, 'graph_input_update_verified': True}
                    if not any(failures):
                        budget()
                        values = agreed('measure_' + variant, lambda: event_samples(
                            operations[variant], args.warmups, args.repetitions))
                        point.update(gpu_ms=values, timing_valid=True)
                    report['points'].append(point)
                    agreed('point_checkpoint', save)
                if any(failures):
                    raise NumericalFailure(next(reason for reason in failures if reason))
                del eager_result, graph_result
            del captured, operations, eager, bound, raw
            agreed('layer_cleanup', lambda: (torch.cuda.synchronize(), gc.collect()))
            print(json.dumps({'rank': rank, 'completed_layer': layer, 'points': len(report['points'])}), flush=True)
        def final_identity():
            require(checkpoint_stats(args.model_path) == initial_stats)
            require(source_sha256() == identity['candidate_source_sha256'])
            require(kernel_source_sha256() == identity['bf16_source_sha256'])
            require(replay_source_sha256() == identity['replay_source_sha256'])
            require(sha256_file(args.capture_dir / 'capture_manifest.json') == identity['capture_manifest_sha256'])
            require(tokenizer_manifest_sha256(args.model_path) == identity['capture_identity']['tokenizer_manifest_sha256'])
            validate_capture_directory(args.capture_dir, identity['capture_identity'])
        agreed('final_identity', final_identity)
        report.update(status='complete', stop_reason='none')
        agreed('validate_complete_report', lambda: validate_rank_report(report, identity, require_complete=True))
        agreed('complete_report', save)
        agreed('complete_progress', lambda: progress('complete'))
    except BudgetStop:
        if report is not None:
            report.update(status='partial', stop_reason='budget')
            save()
        progress('partial')
    except Exception as exc:
        if report is not None:
            report.update(status='failed', stop_reason=exc.reason if isinstance(exc, NumericalFailure) else 'runtime_failure')
            save()
        progress('failed')
        # Never interpolate private values or exception messages into stdout.
        print(json.dumps({'rank': rank, 'status': 'failed', 'error_type': type(exc).__name__}), flush=True)
        return 1
    finally:
        dist.destroy_process_group()
    return 0 if report is not None and report['status'] == 'complete' else 2


def require(condition):
    if not condition:
        raise AssertionError('required invariant failed')


def copy_trace(fixed, trace, sample, layer):
    for destination, name in zip(fixed, ('x', 'ids', 'routing')):
        destination.copy_(trace[name][sample, layer], non_blocking=False)


if __name__ == '__main__':
    sys.exit(main())
