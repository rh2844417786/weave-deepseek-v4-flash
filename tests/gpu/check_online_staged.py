"""Four-H100 online staged gate; original local checkpoint, offline, all 43 layers.

Run with torchrun --nproc_per_node=4 and --model-path /model --output-dir /work/run.
Gloo agrees on local failures before NCCL/IPC stages. This correctness gate may
synchronize and inspect routes; the serving forward does neither.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.checkpoint_identity import checkpoint_identity, checkpoint_stats
from src.weave.ops import PeerRuntime, extension, kernel_source_sha256
from src.weave.staged_plan import build_staged_groups
from src.weave.weights import load_raw_experts
from tests.gpu.check_checkpoint import expert_reference


def atomic_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            os.fchmod(output.fileno(), 0o664)
            json.dump(value, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def local_action(label, operation):
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    failures = [None] * 4
    dist.all_gather_object(failures, error)
    if any(failures):
        raise RuntimeError(f'{label}: rank failures: {failures}')
    return result


def numeric(actual, expected, label):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f'{label}: nonfinite values')
    difference = actual.float() - expected.float()
    relative = (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()
    if relative > .025:
        raise AssertionError(f'{label}: relative L2 {relative} > .025')
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.02, msg=label)
    return {'relative_l2': relative, 'max_abs': difference.abs().max().item()}


def synthetic_raw(rank, device, hidden, intermediate, experts):
    generator = torch.Generator(device=device).manual_seed(271014 + rank)
    local = experts // 4
    shapes = [(local, intermediate, hidden // 2)] * 2 + [(local, hidden, intermediate // 2)]
    shapes += [(local, intermediate, hidden // 32)] * 2 + [(local, hidden, intermediate // 32)]
    return [torch.randint(0 if j < 3 else 120, 256 if j < 3 else 124, shape,
                         dtype=torch.uint8, device=device, generator=generator)
            for j, shape in enumerate(shapes)]


def case_inputs(batch, hidden, experts, variant, device):
    # Identical TP replicated inputs, including dense overflow and empty ranks.
    local = experts // 4
    base = ([0, 1, 2, 3, 4, 5] if variant % 3 == 0 else
            [1, local + 2, 2 * local + 3, 4, local + 5, 2 * local + 6])
    ids = [list(base) if variant % 3 != 2 else
           [(expert + token * 7 + variant) % experts for expert in base]
           for token in range(batch)]
    generator = torch.Generator().manual_seed(2400 + variant)
    x = (torch.randn((batch, hidden), generator=generator) * .125).bfloat16().to(device)
    routing = torch.tensor([[.10, .15, .20, .10, .25, .20]] * batch, device=device)
    # Exercise conversion into the workspace's preallocated int32/FP32 inputs.
    ids_device = torch.tensor(ids, dtype=torch.int64 if variant % 2 else torch.int32, device=device)
    if variant % 2:
        routing = routing.bfloat16()
    return x, ids_device, routing, ids


def local_reference(x, ids, routing, raw, rank, experts, clamp):
    topk, hidden = ids.shape[1], x.shape[1]
    products = torch.zeros((x.shape[0] * topk, hidden), device=x.device)
    local = experts // 4
    for expert in sorted(set(ids.cpu().flatten().tolist())):
        if expert // local != rank:
            continue
        token, slot = torch.where(ids == expert)
        down = expert_reference(x[token], raw, expert % local, clamp)
        products[token * topk + slot] = (
            down.bfloat16() * routing[token, slot].bfloat16().unsqueeze(1)).float()
    return products


def ordered_output(products, batch, hidden, rank):
    # Explicit FP32 slot order; a vectorized reduction can reassociate additions.
    output = torch.zeros((batch, hidden), dtype=torch.float32)
    for slot in range(6):
        output += products[slot::6]
    chunk = (batch + 3) // 4
    start, end = min(batch, rank * chunk), min(batch, (rank + 1) * chunk)
    output[:start].zero_()
    output[end:].zero_()
    return output.bfloat16()


def check_case(ws, persistent, bound, raw, rank, device_group, batch, hidden,
               experts, clamp, variant, layer):
    x, ids, routing, ids_cpu = local_action('case inputs', lambda:
        case_inputs(batch, hidden, experts, variant, torch.device('cuda', rank)))
    expected_local = local_action('independent local reference', lambda:
        local_reference(x, ids, routing, raw, rank, experts, clamp))

    def launch_and_metadata():
        # Poison stale outputs. Empty ranks and shrinking B must overwrite them.
        ws.online_slot_outputs.fill_(float('nan'))
        ws.online_output.fill_(float('nan'))
        ws.run_online(x, ids, routing, bound, clamp)
        torch.cuda.synchronize()
        capacity = ws.online_group_meta.shape[0]
        plan = build_staged_groups(ids_cpu, rank=rank, ep=4, experts=experts)
        expected_meta = list(plan.group_meta) + [(-1, 0)] * (capacity - plan.capacity)
        expected_slots = list(plan.group_slots) + [(-1,) * 16] * (capacity - plan.capacity)
        if ws.online_group_count.cpu().tolist() != [plan.group_count]:
            raise AssertionError('online device group count differs from P1')
        if ws.online_group_meta.cpu().tolist() != [list(row) for row in expected_meta]:
            raise AssertionError('online group metadata differs from P1')
        if ws.online_group_slots.cpu().tolist() != [list(row) for row in expected_slots]:
            raise AssertionError('online group slots differ from P1')
        actual = ws.online_slot_outputs[:batch * 6].clone()
        metrics = numeric(actual, expected_local, 'local routed products')
        remote = (ids.to(torch.int32) // (experts // 4) != rank).flatten()
        if torch.count_nonzero(actual[remote]).item():
            raise AssertionError('nonowned slot contribution was not zero')
        return actual.cpu(), expected_local.cpu(), metrics, plan.group_count

    actual_local, reference_local, local_metrics, group_count = local_action(
        'online launch and metadata', launch_and_metadata)
    all_products = [None] * 4
    dist.all_gather_object(all_products, (actual_local, reference_local))
    exact_global = sum((item[0] for item in all_products), torch.zeros_like(actual_local))
    reference_global = sum((item[1] for item in all_products), torch.zeros_like(reference_local))
    dist.all_reduce(ws.online_slot_outputs[:batch * 6], group=device_group)

    def finalize_and_compare():
        ws.finalize_online(batch)
        torch.cuda.synchronize()
        # Assembly is bit exact relative to the actual computed local slots.
        torch.testing.assert_close(ws.online_slot_outputs[:batch * 6].cpu(), exact_global, rtol=0, atol=0)
        output = ws.online_output[:batch].cpu()
        torch.testing.assert_close(output, ordered_output(exact_global, batch, hidden, rank), rtol=0, atol=0)
        metrics = numeric(output, ordered_output(reference_global, batch, hidden, rank), 'global reference')
        return output, metrics

    output, reference_metrics = local_action('NCCL slot assembly and finalize', finalize_and_compare)
    chunk = (batch + 3) // 4
    start, end = min(batch, rank * chunk), min(batch, (rank + 1) * chunk)
    persistent_bound = local_action('persistent weight binding', lambda: persistent.bind_weights(raw))

    def persistent_compare():
        own = persistent.run(x[start:end], ids[start:end], routing[start:end], persistent_bound,
                             fixed_plan=(1, 1, 0), timeout_seconds=120, clamp=clamp)
        torch.cuda.synchronize()
        expected = torch.zeros((batch, hidden), dtype=torch.bfloat16)
        expected[start:end] = own.cpu()
        return numeric(output, expected, 'persistent cached output')

    persistent_metrics = local_action('persistent numerical comparison', persistent_compare)
    return {'layer': layer, 'batch': batch, 'variant': variant, 'group_count': group_count,
            'local_slots': local_metrics, 'reference': reference_metrics,
            'persistent': persistent_metrics, 'metadata': 'exact_p1_match',
            'assembly': 'exact_fp32_slots_then_ordered_sum_bf16_origin_mask'}


def require_error(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('expected input/foreign bound validation failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    if dist.get_world_size() != 4:
        raise ValueError('online staged gate requires torchrun with four allocated H100 ranks')
    device_group = dist.new_group(backend='nccl', timeout=timedelta(minutes=10))
    device = torch.device('cuda', rank)
    module = local_action('CUDA extension', extension)
    records = []
    identity = [None]
    initial_stats = local_action('checkpoint initial stat', lambda: checkpoint_stats(args.model_path))

    def inspect_checkpoint():
        config = json.loads((args.model_path / 'config.json').read_text())
        shape = tuple(config[key] for key in ('hidden_size', 'moe_intermediate_size',
                      'n_routed_experts', 'num_experts_per_tok', 'num_hidden_layers'))
        if shape != (4096, 2048, 256, 6, 43):
            raise ValueError(f'wrong checkpoint architecture: {shape}')
        if rank == 0:
            identity[0] = checkpoint_identity(args.model_path)
            atomic_report(args.output_dir / 'checkpoint_identity.json', identity[0])
        return float(config['swiglu_limit'])

    clamp = local_action('local checkpoint identity', inspect_checkpoint)
    dist.broadcast_object_list(identity, src=0)
    for synthetic in (True, False):
        hidden, intermediate, experts = (96, 64, 32) if synthetic else (4096, 2048, 256)
        ws = local_action('staged allocation', lambda: module.StagedWorkspace(
            rank, 4, 32, hidden, intermediate, experts, 6))
        persistent = PeerRuntime(None, 32, hidden, intermediate, 6, experts, kernel='tiled')
        pointers = [getattr(ws, name).data_ptr() for name in ('online_slot_outputs', 'online_output',
                    'online_group_meta', 'online_group_slots', 'online_group_count')]
        for layer in ([-1] if synthetic else range(43)):
            raw = local_action('load immutable weights', lambda: (
                synthetic_raw(rank, device, hidden, intermediate, experts) if synthetic else
                load_raw_experts(args.model_path, layer, rank, 4, device=device)))
            bound = local_action('online weight binding', lambda: ws.bind_online_weights(raw))
            if synthetic:
                def validation():
                    foreign = module.StagedWorkspace(rank, 4, 32, hidden, intermediate, experts, 6)
                    x, ids, routing, _ = case_inputs(1, hidden, experts, 0, device)
                    require_error(lambda: foreign.run_online(x, ids, routing, bound, clamp))
                    require_error(lambda: ws.run_online(x.float(), ids, routing, bound, clamp))
                    require_error(lambda: ws.run_online(x, ids, routing, bound, float('nan')))
                    require_error(lambda: ws.finalize_online(0))
                    require_error(lambda: ws.bind_online_weights(raw[:5]))
                local_action('API guards', validation)
            cases = list(enumerate((1, 7, 17, 32, 7, 1))) if synthetic else [(layer, 32)]
            for variant, batch in cases:
                records.append(check_case(ws, persistent, bound, raw, rank, device_group,
                               batch, hidden, experts, clamp, variant, layer))
            if synthetic:
                def binding_lifetime():
                    x, ids, routing, _ = case_inputs(7, hidden, experts, 1, device)
                    ws.run_online(x, ids, routing, bound, clamp)
                    first = ws.online_slot_outputs[:42].clone()
                    # A second layer binding must not overwrite the first handle.
                    zero_raw = [torch.zeros_like(tensor) for tensor in raw]
                    zero_bound = ws.bind_online_weights(zero_raw)
                    del zero_raw
                    gc.collect()
                    ws.run_online(x, ids, routing, zero_bound, clamp)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(ws.online_slot_outputs[:42], torch.zeros_like(first), rtol=0, atol=0)
                    ws.run_online(x, ids, routing, bound, clamp)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(ws.online_slot_outputs[:42], first, rtol=0, atol=0)
                local_action('immutable binding and lifetime', binding_lifetime)
            def stable_addresses():
                after = [getattr(ws, name).data_ptr() for name in ('online_slot_outputs', 'online_output',
                         'online_group_meta', 'online_group_slots', 'online_group_count')]
                if after != pointers:
                    raise AssertionError('online workspace addresses changed')
            local_action('stable workspace storage', stable_addresses)
            del raw, bound
            torch.cuda.synchronize()
            local_action('progress report', lambda: atomic_report(
                args.output_dir / f'online-staged-progress-rank{rank}.json',
                {'status': 'running', 'rank': rank, 'last_layer': layer, 'cases': len(records)}))
            print(json.dumps({'rank': rank, 'online_staged_layer': layer, 'checks': 'passed'}), flush=True)
        resources = local_action('staged resources', ws.resources)
        del ws, persistent
        gc.collect()
    def final_report():
        if checkpoint_stats(args.model_path) != initial_stats:
            raise ValueError('checkpoint changed during the numerical gate')
        report = {'status': 'online_staged_checks_passed', 'rank': rank,
                  'layers': list(range(43)), 'source_sha256': kernel_source_sha256(),
                  'checkpoint_content_sha256': identity[0]['sha256'],
                  'checkpoint_index_sha256': hashlib.sha256(
                      (args.model_path / 'model.safetensors.index.json').read_bytes()).hexdigest(),
                  'cases': records, 'resources': resources, 'rtol': .03, 'atol': .02,
                  'relative_l2_limit': .025, 'cpu_collective': 'gloo',
                  'device_collective': 'nccl_sum_fp32_original_slots',
                  'assembly': 'BF16_rounded_routed_product_FP32_slot_SUM_then_ordered_slots_0_to_5_BF16_origin_mask',
                  'performance': 'not_measured'}
        atomic_report(args.output_dir / f'online-staged-rank{rank}.json', report)
    local_action('final report', final_report)
    dist.destroy_process_group(device_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
