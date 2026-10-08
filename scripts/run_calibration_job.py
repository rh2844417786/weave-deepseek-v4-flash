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
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.container_lifecycle import run_container
from src.weave.runtime_contract import load_contract
from src.weave.calibration import measuring_deadline,atomic_json
from src.weave.experiment_export import export_run


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
    parser.add_argument('--diagnose-case',help='isolated tokens,c,K,mode replay with synchronous CUDA, no profile')
    parser.add_argument('--kernel-timeout-seconds',type=float,default=60.)
    parser.add_argument('--export-dir',type=Path,help='public-safe diagnostic export; default reports/server/<output-name>')
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
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',args.invocation_id): parser.error('safe invocation ID required')
    if not 0<args.kernel_timeout_seconds<=1200: parser.error('kernel timeout must be finite in (0,1200]')
    if args.diagnose_case:
        if args.resume: parser.error('diagnostic replay requires a fresh separate directory')
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error('diagnostic replay requires an empty separate output directory')
        destination=args.export_dir or ROOT/'reports/server'/args.output_dir.name
        if destination.exists(): parser.error('fresh diagnostic export directory required')
    elif args.export_dir: parser.error('export-dir applies only to isolated diagnostic runs')
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
            '-e',f'WEAVE_HOST_HOSTNAME={os.uname().nodename}']
    if args.diagnose_case: docker += ['-e','CUDA_LAUNCH_BLOCKING=1']
    else: docker += ['-e','CUDA_LAUNCH_BLOCKING=0']
    docker += [image,contract['python_executable'],
            '-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            '/workspace/scripts/calibrate_weave.py','--model-path','/model',
            '--output-dir',str(Path('/work')/relatives['output_dir']),
            '--core-run-dir',str(Path('/work')/relatives['core_run_dir']),'--budget-seconds','1200',
            '--kernel-timeout-seconds',str(args.kernel_timeout_seconds)]
    if args.resume: docker.append('--resume')
    if args.diagnose_case: docker += ['--diagnose-case',args.diagnose_case]
    report={'requested_image':args.image,'resolved_image_id':image,'docker_argv':docker,
            'scope':'measured_calibration_not_formal_performance'}
    (args.output_dir/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
    def expired():
        deadline=measuring_deadline(args.output_dir/'progress-rank0.json',args.invocation_id)
        return deadline is not None and time.monotonic()>=deadline
    with (args.output_dir/'calibration.log').open('a') as log:
        lifecycle=run_container(docker,name,args.lifecycle_path,stdout=log,expired=expired)
    if args.diagnose_case:
        status='diagnostic_passed' if lifecycle['exit_code']==0 else ('deadline' if lifecycle['exit_code']==124 else 'failed')
        if status=='diagnostic_passed':
            for rank in range(4):
                file=args.output_dir/f'diagnostic-rank{rank}.json'
                if not file.is_file() or json.loads(file.read_text()).get('status')!='diagnostic_passed':
                    raise ValueError('four passed diagnostic rank reports required')
        summary={'schema_version':1,'scope':'calibration_diagnostic','status':status,
                 'performance':'not_run','used_for_calibration':False,'raw_content_exported':False,
                 'requested_case':args.diagnose_case,'invocation_id':args.invocation_id,
                 'calibration_dir':str(args.output_dir),'core_run_dir':str(args.core_run_dir),
                 'kernel_timeout_seconds':args.kernel_timeout_seconds,'launch_blocking':True,
                 'cleanup':lifecycle,'image_id':image,'export_directory':str(destination)}
        atomic_json(args.output_dir/'experiment.json',summary)
        export_run(args.output_dir,destination)
        print(json.dumps({'status':status,'export_directory':str(destination)}),flush=True)
    return lifecycle['exit_code']


if __name__=='__main__': raise SystemExit(main())
