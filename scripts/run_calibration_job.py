#!/usr/bin/env python3
"""Own an offline four-GPU calibration container under an existing GPUQ lease."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.container_lifecycle import run_container
from src.weave.runtime_contract import load_contract


def main():
    contract=load_contract()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root',type=Path,default=ROOT/'work')
    parser.add_argument('--core-run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--lifecycle-path',type=Path,required=True)
    parser.add_argument('--image',default=contract['image'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--invocation-id',default=uuid.uuid4().hex)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5,6,7':
        parser.error('GPUQ must allocate exactly physical GPUs4,5,6,7')
    if any(not p.is_absolute() for p in (args.model_path,args.work_root,args.core_run_dir,args.output_dir,args.lifecycle_path)):
        parser.error('all model/work/evidence paths must be absolute')
    args.work_root=args.work_root.resolve()
    relatives={}
    for field in ('core_run_dir','output_dir','lifecycle_path'):
        path=getattr(args,field).resolve()
        try: relatives[field]=path.relative_to(args.work_root)
        except ValueError: parser.error('all evidence directories must be under work-root')
        setattr(args,field,path)
    if args.lifecycle_path.exists(): parser.error('fresh lifecycle file required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # No pulling: fail if the user-selected existing image is absent.
    try:
        image=subprocess.run(['docker','image','inspect',args.image,'--format','{{.Id}}'],
            capture_output=True,text=True,check=True,timeout=30).stdout.strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',image): raise ValueError('image content ID missing')
    except BaseException:
        args.lifecycle_path.parent.mkdir(parents=True,exist_ok=True)
        args.lifecycle_path.write_text(json.dumps({'cleanup_status':'not_required','container_started':False})+'\n')
        raise
    name='weave-calibration-'+uuid.uuid4().hex[:12]
    docker=['docker','run','--rm','--pull=never','--name',name,
            '--gpus','"device=4,5,6,7"','--network=host','--shm-size=8g',
            '-v',f'{ROOT}:/workspace:ro','-v',f'{args.work_root}:/work',
            '-v',f'{args.model_path}:/model:ro','-w','/workspace',
            '-e','CUDA_VISIBLE_DEVICES=0,1,2,3','-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
            '-e','GLOO_SOCKET_IFNAME=lo','-e','NCCL_SOCKET_IFNAME=lo','-e','NCCL_IB_DISABLE=1',
            '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
            '-e','HF_DATASETS_OFFLINE=1','-e','TORCH_EXTENSIONS_DIR=/work/torch_extensions',
            '-e','MAX_JOBS=2','-e',f'WEAVE_IMAGE_ID={image}',
            '-e',f'WEAVE_CALIBRATION_INVOCATION_ID={args.invocation_id}',
            '-e',f'WEAVE_IMAGE_REF={args.image}','-e','WEAVE_HOST_ALLOCATION=4,5,6,7',
            '-e',f'WEAVE_HOST_HOSTNAME={os.uname().nodename}',image,contract['python_executable'],
            '-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            '/workspace/scripts/calibrate_weave.py','--model-path','/model',
            '--output-dir',str(Path('/work')/relatives['output_dir']),
            '--core-run-dir',str(Path('/work')/relatives['core_run_dir']),'--budget-seconds','1200']
    if args.resume: docker.append('--resume')
    report={'requested_image':args.image,'resolved_image_id':image,'docker_argv':docker,
            'scope':'measured_calibration_not_formal_performance'}
    (args.output_dir/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output_dir/'calibration.log').open('a') as log:
        lifecycle=run_container(docker,name,args.lifecycle_path,stdout=log)
    return lifecycle['exit_code']


if __name__=='__main__': raise SystemExit(main())
