#!/usr/bin/env python3
"""Offline candidate-only Hopper FP8/BF16 real-trace gate; no service promotion."""
import argparse
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_weave import validate_core_evidence
from scripts.run_model_acceptance import OwnedChild
from src.weave.calibration import atomic_json
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.experiment_export import ensure_public
from src.weave.gpu_allocation import allocated_gpus,GPUAllocation
from src.weave.hopper_candidate import source_manifest,source_sha256
from src.weave.hopper_evidence import VARIANTS,NUMERICAL_POLICY,validate_identity,validate_rank_report,aggregate_rank_reports,replay_source_sha256
from src.weave.ops import kernel_source_sha256,kernel_source_manifest
from src.weave.replay_trace import validate_capture_directory,aggregate_manifests,tokenizer_manifest_sha256
from src.weave.runtime_contract import load_contract


def controller_source_manifest():
    paths=set()
    for directory,pattern in (('scripts','*.py'),('src/weave','*.py'),('tests','*.py'),('configs','*.json')):
        paths.update(p for p in (ROOT/directory).rglob(pattern) if p.is_file())
    return {'schema_version':1,'files':[{'path':p.relative_to(ROOT).as_posix(),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]}


def controller_source_sha256():
    return hashlib.sha256(json.dumps(controller_source_manifest(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def measurement_start(progress,invocation):
    if progress.get('invocation_id')!=invocation:return None
    value=progress.get('measurement_started_monotonic')
    if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
        raise ValueError('invalid measurement clock')
    return value


def docker_command(image,work,model,capture,name,allocation,warmups,repetitions,budget,invocation):
    devices=GPUAllocation.parse(allocation);python=load_contract()['python_executable']
    return ['docker','run','--rm','--pull=never','--name','weave-'+name,
        '--gpus',devices.docker_argument,'--network=host','--shm-size=8g',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work','-v',f'{model}:/model:ro',
        '-v',f'{capture}:/capture:ro','-e',f'CUDA_VISIBLE_DEVICES={devices.container_visible}',
        '-e','GLOO_SOCKET_IFNAME=lo','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '-e','HF_DATASETS_OFFLINE=1','-e','TORCH_EXTENSIONS_DIR=/work/torch_extensions',
        '-e','MAX_JOBS=2','-e','PYTHONUNBUFFERED=1',
        '-e',f'WEAVE_HOST_ALLOCATION={devices.csv}','-e',f'WEAVE_IMAGE_ID={image}',
        '-e',f'WEAVE_IMAGE_REF={load_contract()["image"]}',image,python,
        '/workspace/scripts/check_runtime.py','--stage','native','--output-dir',f'/work/{name}',
        '--exec-command',python,'/workspace/scripts/hopper_candidate_child.py',
        '--model-path','/model','--capture-dir','/capture',
        '--output-dir',f'/work/{name}','--identity-file',f'/work/{name}/identity.json',
        '--warmups',str(warmups),'--repetitions',str(repetitions),'--budget-seconds',str(budget),
        '--invocation-id',invocation]


def export_public(run,destination):
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh public destination required')
    names=['experiment.json','identity.json','runtime.json','container_lifecycle.json','summary.json','oracle-unit-check.json',
        'candidate-source-manifest.json','bf16-source-manifest.json','controller-source-manifest.json',
        'capture-manifest.json']
    names += [f'{prefix}{r}.json' for prefix in ('hopper-rank','hopper-progress-rank') for r in range(4)]
    files={};identity=json.loads((run/'identity.json').read_text()) if (run/'identity.json').is_file() else None
    for name in names:
        path=run/name
        if not path.is_file():continue
        raw=path.read_bytes();value=json.loads(raw)
        if name=='capture-manifest.json':
            if aggregate_manifests(value['ranks'],value['identity'])!=value:
                raise ValueError('capture metadata does not match strict public schema')
        elif name.startswith('hopper-rank'):
            validate_rank_report(value,identity)
        else:ensure_public(value)
        files[name]=raw
    report=json.loads((run/'experiment.json').read_text())
    if report.get('core_run_dir'):
        core=Path(report['core_run_dir'])
        for name in ['runtime.json','checkpoint_identity.json']+[f'{prefix}{r}.json'
                for prefix in ('rank','checkpoint-rank') for r in range(4)]:
            if (core/name).is_file():
                raw=(core/name).read_bytes();ensure_public(json.loads(raw));files['core/'+name]=raw
    private_log=run/'build-and-replay-private.log'
    if private_log.is_file():
        lines=[line for line in private_log.read_text(errors='replace').splitlines()
               if line.lstrip().startswith(('ptxas info','ptxas warning'))]
        files['compilation-resources.txt']=('\n'.join(lines)+'\n').encode()
    destination.mkdir(parents=True)
    manifest={'schema_version':2,'status':report['status'],'numerical_policy':NUMERICAL_POLICY,'raw_content_exported':False,
              'used_for_formal_performance':False,'service_promotion_allowed':False,'files':{}}
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        manifest['files'][name]=hashlib.sha256(raw).hexdigest()
    atomic_json(destination/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work-root','capture-dir','core-run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--warmups',type=int,default=3);parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--budget-seconds',type=float,default=7200.)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    for key in ('work_root','capture_dir','core_run_dir','model_path','export_root'):
        value=getattr(args,key)
        if value is not None:
            if not value.is_absolute():parser.error(f'{key} must be absolute')
            setattr(args,key,value.resolve())
    if not 1<=args.warmups<=20 or not 2<=args.repetitions<=100:
        parser.error('warmups1..20 and repetitions2..100 required')
    if any(not math.isfinite(v) or v<=0 for v in (args.setup_timeout,args.budget_seconds)) or args.budget_seconds>86400:
        parser.error('positive budgets required; measurement at most86400seconds')
    allocation=allocated_gpus();contract=load_contract();invocation=str(uuid.uuid4())
    name='hopper-candidate-'+uuid.uuid4().hex[:12];out=args.work_root/name
    destination=(args.export_root or args.work_root/'public-reports')/name
    command=docker_command(contract['image'],args.work_root,args.model_path,args.capture_dir,name,
                           allocation.csv,args.warmups,args.repetitions,args.budget_seconds,invocation)
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','docker_argv':command,
            'candidate_source_sha256':source_sha256(),'bf16_source_sha256':kernel_source_sha256(),
            'numerical_policy':NUMERICAL_POLICY,
            'service_promotion_allowed':False},indent=2));return 0
    out.mkdir(parents=True,exist_ok=False);owner=OwnedChild();code=1
    report={'schema_version':2,'numerical_policy':NUMERICAL_POLICY,'status':'running','stage':'validating','scope':'Hopper_FP8_local_expert_candidate',
        'started_utc':datetime.now(timezone.utc).isoformat(),'core_run_dir':str(args.core_run_dir),
        'host_gpu_allocation':allocation.csv,'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),
        'invocation_id':invocation,'used_for_formal_performance':False,'service_promotion_allowed':False}
    def save():atomic_json(out/'experiment.json',report)
    previous={sig:signal.signal(sig,owner.signal) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        save()
        candidate_hash=source_sha256();bf16_hash=kernel_source_sha256();replay_hash=replay_source_sha256()
        controller_hash=controller_source_sha256()
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        gate,checkpoint=validate_core_evidence(args.core_run_dir,args.model_path,bf16_hash,{'resolved_image_id':image})
        capture=validate_capture_directory(args.capture_dir)
        if (capture['identity']['checkpoint_content_sha256']!=gate['checkpoint_content_sha256']
                or capture['identity']['image_id']!=image
                or capture['identity']['tokenizer_manifest_sha256']!=tokenizer_manifest_sha256(args.model_path)):
            raise ValueError('archived capture checkpoint/tokenizer/image mismatch')
        capture_file_sha=hashlib.sha256((args.capture_dir/'capture_manifest.json').read_bytes()).hexdigest()
        identity={'candidate_source_sha256':candidate_hash,'bf16_source_sha256':bf16_hash,
            'replay_source_sha256':replay_hash,'checkpoint_content_sha256':gate['checkpoint_content_sha256'],
            'image_id':image,'capture_manifest_sha256':capture_file_sha,'capture_identity':capture['identity'],
            'warmups':args.warmups,'repetitions':args.repetitions,'variants':list(VARIANTS),
            'layers':list(range(43)),'samples':[64,128],'invocation_id':invocation,'numerical_policy':NUMERICAL_POLICY}
        validate_identity(identity);atomic_json(out/'identity.json',identity)
        atomic_json(out/'capture-manifest.json',capture)
        atomic_json(out/'candidate-source-manifest.json',source_manifest())
        atomic_json(out/'bf16-source-manifest.json',kernel_source_manifest())
        atomic_json(out/'controller-source-manifest.json',controller_source_manifest())
        command=docker_command(image,args.work_root,args.model_path,args.capture_dir,name,allocation.csv,
                               args.warmups,args.repetitions,args.budget_seconds,invocation)
        report.update(identity=identity,core_gate=gate,controller_source_sha256=controller_hash,
                      docker_argv=command,stage='gpu_preflight_and_replay');save()
        manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name','weave-'+name,
            '--evidence',str(out/'container_lifecycle.json'),'--log',str(out/'build-and-replay-private.log'),'--']+command
        started=time.monotonic()
        child=owner.spawn(manager,out/'container_lifecycle.json',stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        while child.poll() is None:
            owner.check_cancelled();measurement=None;path=out/'hopper-progress-rank0.json'
            if path.is_file():
                try:measurement=measurement_start(json.loads(path.read_text()),invocation)
                except json.JSONDecodeError:pass
            if measurement is not None and time.monotonic()-measurement>args.budget_seconds+300:
                raise TimeoutError('measurement guard expired')
            if measurement is None and time.monotonic()-started>args.setup_timeout:
                raise TimeoutError('setup/compile/synthetic guard expired')
            time.sleep(.5)
        returncode=child.returncode;owner.stop();owner.check_cancelled();report['child_exit_code']=returncode
        files=[out/f'hopper-rank{r}.json' for r in range(4)]
        if not all(path.is_file() for path in files):raise RuntimeError('four rank reports missing; inspect private build log')
        rows=[json.loads(path.read_text()) for path in files]
        summary=aggregate_rank_reports(rows,identity);atomic_json(out/'summary.json',summary)
        progress=[json.loads((out/f'hopper-progress-rank{r}.json').read_text()) for r in range(4)]
        if any(p.get('invocation_id')!=invocation for p in progress):raise ValueError('progress invocation differs')
        if summary['status']=='complete' and returncode==0 and all(p['stage']=='complete' for p in progress):
            report['status']='compute_candidate_complete'
            code=0
        elif summary['status']=='partial' and all(p['stage']=='partial' for p in progress):
            report['status']='partial';code=124
        else:raise RuntimeError('candidate runtime/structural checks failed; see rank/progress evidence')
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed during run')
        if validate_capture_directory(args.capture_dir)!=capture:raise ValueError('private capture changed during run')
        if hashlib.sha256((args.capture_dir/'capture_manifest.json').read_bytes()).hexdigest()!=capture_file_sha:
            raise ValueError('capture manifest bytes changed during run')
        if tokenizer_manifest_sha256(args.model_path)!=capture['identity']['tokenizer_manifest_sha256']:
            raise ValueError('tokenizer changed during run')
        if (source_sha256()!=candidate_hash or kernel_source_sha256()!=bf16_hash
                or replay_source_sha256()!=replay_hash or controller_source_sha256()!=controller_hash):
            raise ValueError('source changed during run')
        report['stage']='complete' if code==0 else 'partial'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__);code=1
        (out/'controller-error-private.log').write_text(str(exc)+'\n')
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        try:owner.stop()
        except BaseException as exc:report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        export_public(out,destination)
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'work_directory':str(out),'export_directory':str(destination)}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
