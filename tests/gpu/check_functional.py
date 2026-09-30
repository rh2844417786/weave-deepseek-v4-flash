"""Run with torchrun --standalone --nproc-per-node=4; never reports performance."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.ops import PeerRuntime
from src.weave.quantization import decode_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/cuda_functional')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo')
    if dist.get_world_size() != 4:
        raise ValueError('functional gate requires the selected four H100 GPUs')
    device = torch.device('cuda', rank)
    runtime = PeerRuntime(None, 2, 32, 32, 2, 8)

    def matrix(expert):
        packed = bytes([0x32 if expert % 2 == 0 else 0x54] * 16)
        return torch.tensor([decode_row(packed, bytes([127 + row % 2])) for row in range(32)], dtype=torch.bfloat16, device=device)

    weights = [torch.empty((2,32,16),dtype=torch.uint8,device=device) for _ in range(3)]
    scales = [torch.empty((2,32,1),dtype=torch.uint8,device=device) for _ in range(3)]
    for local in range(2):
        for w in weights:
            w[local].fill_(0x32 if (rank*2+local)%2 == 0 else 0x54)
        for s in scales:
            s[local,:,0].copy_(torch.tensor([127+j%2 for j in range(32)],dtype=torch.uint8,device=device))
    raw = tuple(weights+scales)
    x = (torch.arange(1,65,dtype=torch.float32,device=device).reshape(2,32) * .0001 * (rank+1)).to(torch.bfloat16)
    ids = torch.tensor([[(rank+1)%4*2, (rank+1)%4*2+1]]*2, dtype=torch.int32, device=device)
    route_weights = torch.tensor([[.25,.75],[.5,1.]],dtype=torch.float32,device=device)
    valid = torch.tensor([1 if rank != 1 else 0],dtype=torch.int32,device=device)

    def reference(rows):
        outputs = []
        for row in range(rows):
            total = torch.zeros(32,dtype=torch.float32,device=device)
            for slot in range(2):
                w = matrix(int(ids[row,slot]))
                gate = (x[row].float() @ w.float().T).to(torch.bfloat16).float().clamp(max=10)
                up = (x[row].float() @ w.float().T).to(torch.bfloat16).float().clamp(-10,10)
                mid = (torch.nn.functional.silu(gate)*up).to(torch.bfloat16)
                output = (mid.float() @ w.float().T).to(torch.bfloat16).float()
                total += output*route_weights[row,slot]
            outputs.append(total.to(torch.bfloat16))
        return torch.stack(outputs) if outputs else x[:0]

    rows = 1 if rank != 1 else 0
    actual = runtime.run(x,ids,route_weights,raw,valid_count=valid,timeout_seconds=10.)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[:rows],reference(rows),rtol=.025,atol=.002)
    ids.copy_(torch.tensor([[rank*2,rank*2+1]]*2,dtype=torch.int32,device=device))
    valid.fill_(2)
    runtime.run(x,ids,route_weights,raw,valid_count=valid,fixed_plan=(0,1,0),timeout_seconds=10.)
    torch.cuda.synchronize()
    torch.testing.assert_close(runtime.ws.output[:2],reference(2),rtol=.025,atol=.002)
    # Device valid counts and routing buffers are changed between replays.
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runtime.run(x,ids,route_weights,raw,valid_count=valid,timeout_seconds=10.)
    for iteration, rows in enumerate((2,0,1,2)):
        valid.fill_(rows)
        ids.copy_(torch.tensor([[(rank+iteration)%4*2,(rank+iteration)%4*2+1]]*2,dtype=torch.int32,device=device))
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(runtime.ws.output[:rows],reference(rows),rtol=.025,atol=.002)
    # These rates are deliberately synthetic test fixtures, never calibration.
    test_rows=torch.tensor([[1,1,1e10,1e9,1,.5],[1,2,1e10,1e9,1,.5]],dtype=torch.float64,device=device)
    valid.fill_(2)
    runtime.run(x,ids,route_weights,raw,valid_count=valid,profile=test_rows,timeout_seconds=10.)
    torch.cuda.synchronize()
    plan=runtime.ws.schedule.cpu().tolist()
    assert plan[1]==2, plan
    torch.testing.assert_close(runtime.ws.output[:2],reference(2),rtol=.025,atol=.002)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/f'rank{rank}.json').write_text(json.dumps({
        'rank':rank, 'status':'functional_checks_passed', 'core_graph_replay':'passed',
        'sglang_adapter':'not_run', 'full_model':'not_run', 'performance':'not_run',
        'profile':'synthetic_test_fixture_only', 'selected_plan':plan,
    },indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
