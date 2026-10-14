"""Exercise the new eager event ring on all four allocated devices, offline."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.weave.forward_profile import ForwardSampler,validate_forward_profile
from src.weave.execution_profile import profiling_source_sha256
from src.weave.calibration import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import torch
    if torch.cuda.device_count()!=4:raise ValueError('four allocated GPUs required')
    devices=[]
    for rank in range(4):
        torch.cuda.set_device(rank)
        sampler=ForwardSampler(torch.cuda,stride=2,capacity=2)
        a=torch.ones((256,256),device=f'cuda:{rank}',dtype=torch.bfloat16)
        for rows in (32,31,30,29,28):
            ticket=sampler.begin('decode',rows,rows)
            a=torch.mm(a*.001,a*.001)
            sampler.end(ticket)
        torch.cuda.synchronize(rank)
        result=sampler.snapshot();validate_forward_profile(result,stride=2,capacity=2)
        state=result['phases']['decode']
        if (state['valid_rows_total']!=150 or [s['sequence'] for s in state['recent_samples']]!=[2,4]
                or any(s['gpu_stream_ms']<=0 for s in state['recent_samples'])):
            raise ValueError('event ring or row accounting failed on GPU')
        sampler.reset()
        if sampler.snapshot()['phases']['decode']['calls']!=0:raise ValueError('GPU sampler reset failed')
        devices.append({'rank':rank,'status':'passed','sample_gpu_ms':[s['gpu_stream_ms'] for s in state['recent_samples']]})
    atomic_json(args.output,{'status':'passed','scope':'four_device_eager_event_ring_smoke',
        'profiling_source_sha256':profiling_source_sha256(),'devices':devices,'used_for_formal_performance':False})


if __name__=='__main__':main()
