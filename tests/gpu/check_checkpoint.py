"""Independent H4096/I2048 original-MXFP4 EP4 correctness gate (torchrun)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.ops import PeerRuntime, kernel_source_sha256
from src.weave.weights import load_raw_experts
from src.weave.checkpoint_identity import checkpoint_identity, checkpoint_stats

H, I, E, TOPK = 4096, 2048, 256, 6


def dequantize(packed, scales):
    # Independent tensor implementation; do not call the extension's decoder.
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    values = lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1., 1.)
    exponents = scales.int() - 127
    values = torch.ldexp(values.reshape(packed.shape[0], -1, 32), exponents.unsqueeze(-1))
    if (scales == 255).any() or not torch.isfinite(values).all():
        raise ValueError('nonfinite original expert scale/weight')
    return values.reshape(packed.shape[0], -1)


def expert_reference(x, raw, local_id, clamp):
    # FP32 matmul; native BF16 storage at gate/up/SiLU/product/down boundaries.
    w1, w3, w2 = [dequantize(raw[j][local_id], raw[j+3][local_id]) for j in range(3)]
    gate = (x.float() @ w1.T).bfloat16().float().clamp(max=clamp)
    up = (x.float() @ w3.T).bfloat16().float().clamp(-clamp, clamp)
    activation = torch.nn.functional.silu(gate.bfloat16())
    mid = (activation * up.bfloat16()).float()
    return (mid @ w2.T).bfloat16().float()


def distributed_reference(all_inputs, raw, rank, clamp):
    parts = []
    for item in all_inputs:
        x, ids, routing, rows = item
        x = x.cuda(rank)
        result = torch.zeros((rows, H), device=x.device)
        # Group by expert; each original tensor is decoded once per origin.
        for expert in sorted(set(ids[:rows].flatten().tolist())):
            if expert // 64 != rank:
                continue
            token, slot = torch.where(ids[:rows] == expert)
            outputs = expert_reference(x[token], raw, expert % 64, clamp)
            routed = (outputs.bfloat16() * routing[token, slot].cuda(rank).bfloat16().unsqueeze(1)).float()
            result.index_add_(0, token.cuda(rank), routed)
        parts.append(result.cpu())
    gathered = [None] * 4
    dist.all_gather_object(gathered, parts)
    # Sum already-rounded BF16 contributions in FP32. This oracle checks the
    # Weave global combine, not native per-rank/shared-expert all-reduce order.
    return sum((r[rank] for r in gathered), torch.zeros_like(parts[rank])).bfloat16().cuda(rank)


def synthetic_raw(rank, device):
    gen = torch.Generator(device=device).manual_seed(9173 + rank*101)
    weights = [torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen, device=device)
               for shape in [(64, I, H//2), (64, I, H//2), (64, H, I//2)]]
    scales = [torch.randint(120, 124, shape, dtype=torch.uint8, generator=gen, device=device)
              for shape in [(64, I, H//32), (64, I, H//32), (64, H, I//32)]]
    return tuple(weights + scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layers', default='all', help='all 43, or comma-separated checkpoint layer IDs')
    parser.add_argument('--kernels', default='scalar,wmma,tiled')
    args = parser.parse_args()
    layers = list(range(43)) if args.layers == 'all' else [int(v) for v in args.layers.split(',')]
    kernels = args.kernels.split(',')
    if not layers or len(set(layers)) != len(layers) or any(v not in range(43) for v in layers):
        parser.error('layers must be distinct IDs in 0..42')
    if not kernels or any(v not in ('scalar', 'wmma', 'tiled') for v in kernels):
        parser.error('kernels must be scalar,wmma,tiled or a subset')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo')
    if dist.get_world_size() != 4:
        raise ValueError('checkpoint gate requires four allocated H100 ranks')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = [None]
    initial_stats = checkpoint_stats(args.model_path)
    if rank == 0:
        identity[0] = checkpoint_identity(args.model_path, progress=lambda n, total, name:
            print(json.dumps({'checkpoint_hash_progress': [n, total], 'file': name}), flush=True))
        (args.output_dir/'checkpoint_identity.json').write_text(json.dumps(identity[0], indent=2)+'\n')
    dist.broadcast_object_list(identity, src=0)
    device = torch.device('cuda', rank)
    runtime = PeerRuntime(None, 32, H, I, TOPK, E)
    gen = torch.Generator(device=device).manual_seed(542 + rank)
    x_all = (torch.randn((32, H), generator=gen, device=device)*.125).bfloat16()
    x = x_all[:8]
    ids = torch.tensor([[((rank+s)%4)*64+(t*7+s*9)%64 for s in range(TOPK)]
                        for t in range(8)], device=device, dtype=torch.int32)
    routing = torch.tensor([[.1, .15, .2, .1, .25, .2]]*8, device=device)
    valid = torch.tensor([8 if rank != 1 else 0], device=device, dtype=torch.int32)
    props = torch.cuda.get_device_properties(rank)
    plans = [('remote_K2', (1, 2, 0)), ('steal_K2', (props.multi_processor_count-1, 2, 1))]
    config = json.loads((args.model_path/'config.json').read_text())
    architecture = (config['hidden_size'], config['moe_intermediate_size'], config['n_routed_experts'],
                    config['num_experts_per_tok'], config['num_hidden_layers'])
    if architecture != (H, I, E, TOPK, 43):
        raise ValueError(f'wrong checkpoint architecture: {architecture}')
    clamp = float(config['swiglu_limit'])
    records = []

    def compare(expected, rows, tag):
        actual = runtime.ws.output[:rows].clone()
        if not torch.isfinite(actual).all():
            raise AssertionError(f'{tag}: nonfinite output')
        error = (actual.float()-expected.float())
        norm = (error.norm()/expected.float().norm().clamp_min(1e-12)).item()
        metrics = {'relative_l2': norm, 'max_abs': error.abs().max().item() if rows else 0.}
        if norm > .025:
            raise AssertionError(f'{tag}: relative L2 {norm} > .025')
        torch.testing.assert_close(actual, expected, rtol=.03, atol=.02, msg=tag)
        return metrics

    for layer in [-1] + layers:
        raw = synthetic_raw(rank, device) if layer == -1 else load_raw_experts(args.model_path, layer, rank, 4, device=device)
        # Identity evidence for all local experts and all three projections.
        digests = [hashlib.sha256(t[:, :1].cpu().numpy().tobytes()).hexdigest() for t in raw]
        identities = [None]*4
        dist.all_gather_object(identities, digests)
        if layer == -1 and len({tuple(v) for v in identities}) != 4:
            raise AssertionError('synthetic expert identity oracle is not rank-distinct')
        rows = 8 if rank != 1 else 0
        valid.fill_(rows)
        inputs = [None]*4
        dist.all_gather_object(inputs, (x.cpu(), ids.cpu(), routing.cpu(), rows))
        expected = distributed_reference(inputs, raw, rank, clamp)
        for kernel in kernels:
            runtime.kernel = kernel
            for label, plan in plans:
                runtime.run(x, ids, routing, raw, valid_count=valid, fixed_plan=plan, timeout_seconds=120, clamp=clamp)
                torch.cuda.synchronize()
                metrics = compare(expected, rows, f'layer{layer}/{kernel}/{label}')
                stolen, computed = runtime.ws.work_counters.cpu().tolist()
                workload = runtime.ws.workload.cpu().tolist()
                tile_counts = runtime.ws.tile_counters.cpu().tolist()
                if computed != workload[4]:
                    raise AssertionError('not every expert task completed')
                if kernel == 'tiled' and (tile_counts[0] != workload[6] or tile_counts[1] != stolen):
                    raise AssertionError('tile completion/stealing counters differ from planned work')
                actual_counts = [None]*4
                dist.all_gather_object(actual_counts, (stolen, computed))
                if label == 'steal_K2' and sum(v[0] for v in actual_counts) == 0:
                    raise AssertionError('nonzero stealing budget did not produce any completed stolen task')
                records.append({'layer': layer, 'kernel': kernel, 'case': label, 'metrics': metrics,
                                'completed_stolen': stolen, 'completed_tasks': computed,
                                'workload': workload, 'tile_counters': tile_counts})
            # Capture once, then replay with changed routes/counts. Some ranks
            # become empty; other ranks' expert identities change between runs.
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runtime.run(x, ids, routing, raw, valid_count=valid, fixed_plan=(1, 2, 0), timeout_seconds=120, clamp=clamp)
            original = ids.clone()
            for replay in range(2):
                rows = (rank+replay)%3
                valid.fill_(rows)
                ids.copy_((original + (replay+1)*17) % E)
                # Route mutation precedes replay on every rank.
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                dist.all_gather_object(inputs, (x.cpu(), ids.cpu(), routing.cpu(), rows))
                expected_replay = distributed_reference(inputs, raw, rank, clamp)
                metrics = compare(expected_replay, rows, f'layer{layer}/{kernel}/replay{replay}')
                records.append({'layer': layer, 'kernel': kernel, 'case': f'graph_replay{replay}', 'metrics': metrics})
            ids.copy_(original)
            rows = 8 if rank != 1 else 0
            valid.fill_(rows)
        if layer == -1 and 'tiled' in kernels:
            # Exercise full M16 groups, not just padded decode-sized rows.
            grouped_ids = ids[:1].repeat(32, 1)
            grouped_routing = routing[:1].repeat(32, 1)
            grouped_rows = 32 if rank != 1 else 0
            grouped_valid = torch.tensor([grouped_rows], device=device, dtype=torch.int32)
            grouped_inputs = [None]*4
            dist.all_gather_object(grouped_inputs, (x_all.cpu(), grouped_ids.cpu(), grouped_routing.cpu(), grouped_rows))
            grouped_expected = distributed_reference(grouped_inputs, raw, rank, clamp)
            runtime.kernel = 'tiled'
            for chunks in (1, 2, 4, 8):
                runtime.run(x_all, grouped_ids, grouped_routing, raw, valid_count=grouped_valid,
                            fixed_plan=(1, chunks, 1), timeout_seconds=120, clamp=clamp)
                torch.cuda.synchronize()
                metrics = compare(grouped_expected, grouped_rows, f'full_group_rows/K{chunks}')
                workload = runtime.ws.workload.cpu().tolist()
                completed, stolen = runtime.ws.tile_counters.cpu().tolist()
                if completed != workload[6] or workload[5] == 0 or workload[5] >= workload[4]:
                    raise AssertionError('same-expert tokens were not grouped into shared tiles')
                records.append({'layer': -1, 'kernel': 'tiled', 'case': f'grouped_rows_K{chunks}',
                                'metrics': metrics, 'workload': workload,
                                'tile_counters': [completed, stolen]})
        del raw
        torch.cuda.synchronize()
        print(json.dumps({'rank': rank, 'checkpoint_layer': layer, 'checks': 'passed'}), flush=True)
        (args.output_dir/f'checkpoint-progress-rank{rank}.json').write_text(json.dumps(records, indent=2)+'\n')
    report = {'status': 'checkpoint_checks_passed', 'rank': rank, 'layers': layers, 'kernels': kernels,
              'checkpoint_content_sha256': identity[0]['sha256'],
              'architecture': list(architecture), 'source_sha256': kernel_source_sha256(),
              'checkpoint_index_sha256': hashlib.sha256((args.model_path/'model.safetensors.index.json').read_bytes()).hexdigest(),
              'expert_identity_digests_scope': 'first_row_all_local_experts_per_raw_tensor',
              'last_layer_identity_digests': digests, 'cases': records,
              'sglang_graph': 'not_checked_by_core_gate', 'performance': 'not_measured'}
    if checkpoint_stats(args.model_path) != initial_stats:
        raise ValueError('checkpoint changed during the numerical gate')
    (args.output_dir/f'checkpoint-rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
