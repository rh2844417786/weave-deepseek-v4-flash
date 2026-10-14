"""Single allocated H100 staged-kernel edge and CUDA graph correctness gate.

Uses small synthetic original-format MXFP4 weights and the independent tensor
oracle from check_checkpoint.py. Real 43-layer coverage belongs to replay.
No downloads, timings, or performance claims are made by this gate.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.ops import extension, kernel_source_sha256
from src.weave.staged_plan import build_staged_groups
from tests.gpu.check_checkpoint import expert_reference


def require_error(operation, label):
    try:
        operation()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f'{label}: expected validation failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda', args.device)
    module = extension()
    gen = torch.Generator(device=device).manual_seed(241014)
    records = []
    for batch, hidden, intermediate in ((1, 32, 32), (17, 96, 96), (32, 128, 64)):
        rank, ep, experts, topk = 0, 2, 8, 6
        ws = module.StagedWorkspace(rank, ep, batch, hidden, intermediate, experts, topk)
        shapes = ((4, intermediate, hidden // 2), (4, intermediate, hidden // 2),
                  (4, hidden, intermediate // 2), (4, intermediate, hidden // 32),
                  (4, intermediate, hidden // 32), (4, hidden, intermediate // 32))
        raw = [torch.randint(0 if j < 3 else 120, 256 if j < 3 else 124, shape,
                             dtype=torch.uint8, device=device, generator=gen)
               for j, shape in enumerate(shapes)]
        for bad in ([raw[0].float()] + raw[1:], [raw[0][..., :1]] + raw[1:],
                    [raw[0].cpu()] + raw[1:], raw[:5]):
            require_error(lambda bad=bad: ws.bind_weights(bad), 'raw weight contract')
        ws.bind_weights(raw)
        x = (torch.randn((batch, hidden), generator=gen, device=device) * .2).bfloat16()
        ids_cpu = [[2, 0, 4, 1, 6, 7]] * batch
        ids = torch.tensor(ids_cpu, dtype=torch.int32, device=device)
        routing = torch.tensor([[.13, .37, .09, .21, .08, .12]] * batch, device=device)
        plan = build_staged_groups(ids_cpu, rank=rank, ep=ep, experts=experts,
                                   topk=topk, capacity=batch * topk)
        meta = torch.tensor(plan.group_meta, dtype=torch.int32, device=device)
        slots = torch.tensor(plan.group_slots, dtype=torch.int32, device=device)
        count = torch.tensor([plan.group_count], dtype=torch.int32, device=device)
        expected_slots = torch.zeros((batch * topk, hidden), device=device)
        for expert in (0, 1, 2):
            token, slot = torch.where(ids == expert)
            expected_slots[token * topk + slot] = expert_reference(x[token], raw, expert, 10.)
        expected = torch.zeros((batch, hidden), device=device)
        for slot in range(topk):
            expected += (expected_slots[slot::topk].bfloat16()
                         * routing[:, slot:slot + 1].bfloat16()).float()
        # Strong references in bind_weights must keep all six tensors alive.
        del raw
        gc.collect()
        output_ptr, slots_ptr = ws.output.data_ptr(), ws.slot_outputs.data_ptr()
        for grid in (132, 264, 528, 0):
            ws.run(x, ids, routing, meta, slots, count, grid, 10.)
            torch.cuda.synchronize()
            torch.testing.assert_close(ws.slot_outputs, expected_slots, rtol=.03, atol=.02)
            torch.testing.assert_close(ws.output, expected, rtol=.03, atol=.02)
            eager = ws.output.clone()
            ws.bind_inputs(x, ids, routing, meta, slots, count)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ws.run_gate_up(grid, 10.)
                ws.run_down(grid)
                ws.run_reduce()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(ws.output, eager, rtol=0, atol=0)
            if ws.output.data_ptr() != output_ptr or ws.slot_outputs.data_ptr() != slots_ptr:
                raise AssertionError('workspace output addresses changed')
            records.append({'batch': batch, 'h': hidden, 'i': intermediate, 'grid': grid,
                            'eager': 'passed', 'graph': 'passed'})
        require_error(lambda: ws.run_gate_up(133), 'unsupported grid')
        require_error(lambda: ws.run_gate_up(132, float('nan')), 'invalid clamp')
        require_error(lambda: ws.bind_inputs(x.float(), ids, routing, meta, slots, count),
                      'input dtype')
    # A genuinely empty rank, followed by graph replay that changes device-side
    # group_count. The frozen capacity stays constant for direct mapping.
    ws = module.StagedWorkspace(0, 2, 1, 32, 32, 8, 2)
    raw = [torch.zeros(shape, dtype=torch.uint8, device=device)
           for shape in ((4, 32, 16),) * 3 + ((4, 32, 1),) * 3]
    ws.bind_weights(raw)
    x = torch.ones((1, 32), dtype=torch.bfloat16, device=device)
    ids = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    routing = torch.tensor([[.5, .5]], device=device)
    plan = build_staged_groups([[0, 1]], rank=0, ep=2, experts=8, topk=2)
    meta = torch.tensor(plan.group_meta, dtype=torch.int32, device=device)
    slots = torch.tensor(plan.group_slots, dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    ws.run(x, ids, routing, meta, slots, count, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.run_gate_up(0)
        ws.run_down(0)
        ws.run_reduce()
    count.zero_()
    ids.copy_(torch.tensor([[4, 5]], dtype=torch.int32, device=device))
    # Deliberately poison stale outputs; remote slots must become zero.
    ws.slot_outputs.fill_(float('nan'))
    ws.output.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(ws.output, torch.zeros_like(ws.output), rtol=0, atol=0)
    torch.testing.assert_close(ws.slot_outputs, torch.zeros_like(ws.slot_outputs), rtol=0, atol=0)
    report = {'status': 'synthetic_staged_checks_passed', 'source_sha256': kernel_source_sha256(),
              'cases': records, 'empty_rank_graph': 'passed', 'resources': ws.resources(),
              'real_checkpoint': 'not_checked_here', 'performance': 'not_measured'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
