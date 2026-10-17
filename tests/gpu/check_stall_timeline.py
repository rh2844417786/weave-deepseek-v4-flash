"""Synthetic four-rank CUDA collector smoke; run before model smoke."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    result={'rank':rank,'status':'failed','instrumented_diagnostic':True,'used_for_formal_performance':False,'numerical_policy':'record_only'}
    try:
        import torch
        from src.weave.stall_timeline import StallTimeline,TorchBackend
        from src.weave.stall_hooks import stall_source_sha256
        if int(os.environ['WORLD_SIZE'])!=4: raise RuntimeError('requires four ranks')
        torch.cuda.set_device(local)
        timeline=StallTimeline(TorchBackend(torch,local),rank=rank,device=local,
            source_sha256=stall_source_sha256(),identity={'image':os.environ.get('WEAVE_IMAGE_ID','existing-image'),'candidate':'synthetic'})
        value=torch.ones((64,64),device='cuda')
        # Exercise the HTTP control thread's device/default-stream context without
        # opening ports. Future.result propagates any control-thread exception.
        def control(operation):
            with torch.cuda.device(local):
                return operation()
        stream=torch.cuda.Stream(device=local)
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(control,timeline.arm).result()
            with torch.cuda.stream(stream):
                for phase in ['prefill']+['decode']*33:
                    with timeline.forward(phase,64):
                        for layer in range(43):
                            with timeline.routed(layer): value=value+1
            report=worker.submit(control,timeline.snapshot).result()
        if report['failures'] or len(report['calls'])!=34 or len(report['samples'])!=3: raise RuntimeError('collector coverage failed')
        for sample in report['samples']:
            if not math.isfinite(sample['gpu_duration_ms']) or sample['gpu_duration_ms']<=0: raise RuntimeError('invalid direct model duration')
            if not all(math.isfinite(sample[k]) for k in ('gpu_start_ms','gpu_end_ms')): raise RuntimeError('invalid model offsets')
            previous=sample['gpu_start_ms']
            if len(sample['routed'])!=43: raise RuntimeError('routed coverage failed')
            for layer in sample['routed']:
                if not all(math.isfinite(layer[k]) for k in ('start_ms','end_ms','duration_ms')) or layer['duration_ms']<=0: raise RuntimeError('invalid direct routed duration')
                if not previous<=layer['start_ms']<=layer['end_ms']: raise RuntimeError('CUDA event order failed')
                previous=layer['end_ms']
            if previous>sample['gpu_end_ms']: raise RuntimeError('CUDA model event order failed')
        if not torch.isfinite(value).all().item(): raise RuntimeError('nonfinite synthetic output')
        result.update(status='passed',collector=report)
    except BaseException as error:
        private=output/f'stall-smoke-private-rank{rank}.traceback'
        private.write_text(traceback.format_exc()); private.chmod(0o600)
        result.update(error=type(error).__name__,reason='collector smoke failed; see private traceback')
    destination=output/f'stall-smoke-rank{rank}.json'
    if destination.exists(): raise RuntimeError('fresh smoke output required')
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); temporary.chmod(0o664); temporary.replace(destination)
    if result['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
