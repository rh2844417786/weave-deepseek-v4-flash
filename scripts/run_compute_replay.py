#!/usr/bin/env python3
"""Offline P0/P1 replay of validated private real C32 traces, with resumable points."""
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
from src.weave.gpu_allocation import allocated_gpus,GPUAllocation
from src.weave.ops import kernel_source_sha256,kernel_source_manifest
from src.weave.replay_benchmark import ReplayStore,VARIANTS,paired_summary,public_replay,digest
from src.weave.replay_trace import validate_capture_directory,tokenizer_manifest_sha256,aggregate_manifests
from src.weave.runtime_contract import load_contract


def replay_source_manifest():
    # Bind the runtime/probe/gate dependency closure, not just the timed loop.
    paths=set()
    for directory,pattern in (('scripts','*.py'),('src/weave','*.py'),('tests/gpu','*.py'),('configs','*.json')):
        paths.update(p for p in (ROOT/directory).rglob(pattern) if p.is_file())
    return {'schema_version':1,'files':[{ 'path':p.relative_to(ROOT).as_posix(),
             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]}


def replay_source_sha256():return digest(replay_source_manifest())


def measurement_start(progress,invocation):
    if progress.get('invocation_id')!=invocation:return None
    value=progress.get('measurement_started_monotonic')
    if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
        raise ValueError('invalid replay measurement clock')
    return value


def docker_argv(image,work,model,capture,run_name,allocation,warmups,repetitions,stage_repetitions,budget,timeout,invocation='preview'):
    allocation=GPUAllocation.parse(allocation)
    return ['docker','run','--rm','--pull=never','--name','weave-'+run_name,
        '--gpus',allocation.docker_argument,'--network=host','--shm-size=8g',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work','-v',f'{model}:/model:ro','-v',f'{capture}:/trace:ro',
        '-e',f'CUDA_VISIBLE_DEVICES={allocation.container_visible}','-e','GLOO_SOCKET_IFNAME=lo',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','TORCH_EXTENSIONS_DIR=/work/torch_extensions','-e','MAX_JOBS=2',
        '-e',f'WEAVE_HOST_ALLOCATION={allocation.csv}','-e',f'WEAVE_IMAGE_ID={image}',
        '-e',f'WEAVE_IMAGE_REF={load_contract()["image"]}','-e','PYTHONUNBUFFERED=1',
        image,load_contract()['python_executable'],'/workspace/scripts/check_runtime.py',
        '--stage','native','--output-dir',f'/work/{run_name}','--exec-command',load_contract()['python_executable'],
        '/workspace/scripts/compute_replay_child.py','--output-dir',f'/work/{run_name}',
        '--model-path','/model','--capture-dir','/trace','--identity-file',f'/work/{run_name}/identity.json',
        '--warmups',str(warmups),'--repetitions',str(repetitions),'--stage-repetitions',str(stage_repetitions),
        '--budget-seconds',str(budget),'--kernel-timeout',str(timeout),'--invocation-id',invocation]


def export_replay(run,destination):
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh public export required')
    files={}
    names=['experiment.json','identity.json','runtime.json','container_lifecycle.json','capture_manifest.json',
           'source_manifest.json','replay-source-manifest.json','summary.json','staged-synthetic.json']
    names += [f'{prefix}-rank{rank}.json' for prefix in ('replay','resources','progress') for rank in range(4)]
    for name in names:
        file=run/name
        if file.is_file():
            raw=file.read_bytes();value=json.loads(raw)
            if name=='capture_manifest.json':
                # x/ids/routing here are exact shape/dtype descriptors, never values.
                if aggregate_manifests(value['ranks'],value['identity'])!=value:
                    raise ValueError('public capture metadata differs from exact descriptor schema')
            else:public_replay(value)
            files[name]=raw
    report=json.loads((run/'experiment.json').read_text())
    if report.get('core_run_dir'):
        core=Path(report['core_run_dir'])
        for name in ['runtime.json','checkpoint_identity.json']+[f'{prefix}{r}.json'
                for prefix in ('rank','checkpoint-rank') for r in range(4)]:
            file=core/name
            if file.is_file():
                raw=file.read_bytes();public_replay(json.loads(raw));files['core/'+name]=raw
        # Replay normally reuses the extension compiled by this core gate.
        # Preserve its compiler resource evidence without exporting the raw log.
        build_log=core/'build-and-test.log'
        if build_log.is_file():
            lines=[line for line in build_log.read_text(errors='replace').splitlines()
                   if line.lstrip().startswith(('ptxas info','ptxas warning'))]
            files['core/compilation-resources.txt']=('\n'.join(lines)+'\n').encode()
    log=run/'build-and-replay.log'
    if log.exists():
        history=run/'compilation-resources-history.txt'
        lines=(history.read_text().splitlines() if history.exists() else [])
        lines += [line for line in log.read_text(errors='replace').splitlines() if line.lstrip().startswith(('ptxas info','ptxas warning'))]
        files['compilation-resources.txt']=('\n'.join(lines)+'\n').encode()
    destination.mkdir(parents=True)
    manifest={'schema_version':1,'status':report['status'],'used_for_formal_performance':False,'files':{}}
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        manifest['files'][name]=hashlib.sha256(raw).hexdigest()
    atomic_json(destination/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--core-run-dir',type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--resume-dir',type=Path)
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--warmups',type=int,default=3)
    parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--stage-repetitions',type=int,default=3)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--budget-seconds',type=float,default=7200.)
    parser.add_argument('--kernel-timeout',type=float,default=120.)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    for name in ('work_root','capture_dir','core_run_dir','model_path','resume_dir','export_root'):
        path=getattr(args,name)
        if path is not None:
            if not path.is_absolute():parser.error(f'{name} must be absolute')
            setattr(args,name,path.resolve())
    if not 1<=args.warmups<=20 or not 2<=args.repetitions<=100 or not 1<=args.stage_repetitions<=10:
        parser.error('bounded warmups/repetitions required')
    if any(not math.isfinite(v) or v<=0 for v in (args.setup_timeout,args.budget_seconds,args.kernel_timeout)):
        parser.error('finite positive timeouts required')
    if args.kernel_timeout>1200:parser.error('kernel timeout exceeds device guard limit')
    allocation=allocated_gpus();contract=load_contract()
    invocation=uuid.uuid4().hex
    out=args.resume_dir or args.work_root/('compute-replay-'+uuid.uuid4().hex[:12])
    if args.resume_dir:
        if out.parent!=args.work_root or not out.name.startswith('compute-replay-'):
            parser.error('resume-dir must be a previous compute-replay-* directly under work-root')
    command=docker_argv(contract['image'],args.work_root,args.model_path,args.capture_dir,out.name,
                        allocation.csv,args.warmups,args.repetitions,args.stage_repetitions,args.budget_seconds,args.kernel_timeout,invocation)
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','docker_argv':command},indent=2));return 0
    args.work_root.mkdir(parents=True,exist_ok=True)
    if args.resume_dir:
        if not out.is_dir():parser.error('resume directory missing')
    else:out.mkdir()
    destination=(args.export_root or args.work_root/'public-reports')/(out.name+'-'+uuid.uuid4().hex[:8])
    report={'schema_version':1,'status':'running','scope':'p0_p1_frozen_trace_compute_controls',
        'measurement_mode':'frozen_trace_microbenchmark','used_for_formal_performance':False,
        'started_utc':datetime.now(timezone.utc).isoformat(),'stage':'validating','core_run_dir':str(args.core_run_dir),
        'private_trace_directory':str(args.capture_dir),'host_gpu_allocation':allocation.csv,
        'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),'resume':bool(args.resume_dir),'invocation_id':invocation,
        'source_sha256':kernel_source_sha256(),'replay_source_sha256':replay_source_sha256()}
    owner=OwnedChild();code=1
    def save():atomic_json(out/'experiment.json',report)
    def cancel(number,frame):owner.signal(number,frame)
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        save()
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            capture_output=True,text=True,timeout=30,check=True).stdout.strip()
        gate,checkpoint=validate_core_evidence(args.core_run_dir,args.model_path,report['source_sha256'],{'resolved_image_id':image})
        capture=validate_capture_directory(args.capture_dir)
        if (capture['identity']['checkpoint_content_sha256']!=gate['checkpoint_content_sha256']
                or capture['identity']['image_id']!=image
                or capture['identity']['tokenizer_manifest_sha256']!=tokenizer_manifest_sha256(args.model_path)):
            raise ValueError('frozen trace checkpoint/tokenizer/image differs from current contract')
        identity={'schema_version':1,'source_sha256':report['source_sha256'],
            'replay_source_sha256':report['replay_source_sha256'],'image_id':image,
            'checkpoint_content_sha256':gate['checkpoint_content_sha256'],
            'capture_manifest_sha256':digest(capture),'capture_identity':capture['identity'],
            'sample_sequences':capture['config']['sample_sequences'],'layers':list(range(43)),
            'variants':list(VARIANTS),'warmups':args.warmups,'repetitions':args.repetitions,
            'stage_repetitions':args.stage_repetitions,'kernel_timeout_seconds':args.kernel_timeout}
        if args.resume_dir and json.loads((out/'identity.json').read_text())!=identity:
            raise ValueError('resume source/config/trace identity differs; choose fresh output')
        atomic_json(out/'identity.json',identity);atomic_json(out/'capture_manifest.json',capture)
        atomic_json(out/'source_manifest.json',kernel_source_manifest())
        atomic_json(out/'replay-source-manifest.json',replay_source_manifest())
        command=docker_argv(image,args.work_root,args.model_path,args.capture_dir,out.name,
            allocation.csv,args.warmups,args.repetitions,args.stage_repetitions,args.budget_seconds,args.kernel_timeout,invocation)
        if args.resume_dir and (out/'build-and-replay.log').exists():
            history=out/'compilation-resources-history.txt'
            prior=history.read_text() if history.exists() else ''
            lines=[line for line in (out/'build-and-replay.log').read_text(errors='replace').splitlines()
                   if line.lstrip().startswith(('ptxas info','ptxas warning'))]
            history.write_text(prior+'\n'.join(lines)+'\n')
        report.update(resolved_image_id=image,core_gate=gate,docker_argv=command,stage='running_gpu_controls');save()
        manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name','weave-'+out.name,
            '--evidence',str(out/'container_lifecycle.json'),'--log',str(out/'build-and-replay.log'),'--']+command
        started=time.monotonic()
        process=owner.spawn(manager,out/'container_lifecycle.json',stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        while process.poll() is None:
            owner.check_cancelled();marker=out/'progress-rank0.json';started_measurement=None
            if marker.exists():
                try:started_measurement=measurement_start(json.loads(marker.read_text()),invocation)
                except json.JSONDecodeError:pass
            if started_measurement is not None and time.monotonic()-started_measurement>args.budget_seconds+args.kernel_timeout+60:
                report['controller_timeout']='measurement_guard';owner.stop();owner.check_cancelled();break
            if started_measurement is None and time.monotonic()-started>args.setup_timeout:
                report['controller_timeout']='setup_guard';owner.stop();owner.check_cancelled();break
            time.sleep(.5)
        returncode=process.returncode;owner.stop();owner.check_cancelled()
        report['child_exit_code']=returncode
        files=[out/f'replay-rank{r}.json' for r in range(4)]
        if all(file.exists() for file in files):
            ranks=[json.loads(file.read_text()) for file in files]
            for rank,row in enumerate(ranks):
                if row['identity']['rank']!=rank or any(row['identity'].get(k)!=v for k,v in identity.items()):
                    raise ValueError('rank replay identity differs')
                ReplayStore(files[rank],row['identity'])
            progress=[json.loads((out/f'progress-rank{r}.json').read_text()) for r in range(4)]
            current=all(p.get('invocation_id')==invocation for p in progress)
            if all(row['status']=='complete' for row in ranks) and returncode==0 and current and all(p['stage']=='complete' for p in progress):
                expected={f'{layer}:{sample}:{variant}' for layer in range(43) for sample in range(2) for variant in VARIANTS}
                if any(ReplayStore(files[r],ranks[r]['identity']).keys()!=expected for r in range(4)):
                    raise ValueError('all43/sample/variant cases required')
                atomic_json(out/'summary.json',paired_summary(ranks));report['status']='compute_controls_complete';code=0
            elif all(row['status'] in ('partial','complete') for row in ranks) and current and (
                    (returncode==0 and all(p['stage']=='partial' for p in progress)) or report.get('controller_timeout')):
                report['status']='partial';code=124
            else:raise RuntimeError('GPU replay rank failed; inspect progress and private log')
        else:raise RuntimeError('GPU replay ended without four rank reports; inspect private build log')
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed during replay')
        if validate_capture_directory(args.capture_dir)!=capture:raise ValueError('private trace changed during replay')
        if kernel_source_sha256()!=report['source_sha256'] or replay_source_sha256()!=report['replay_source_sha256']:
            raise ValueError('replay source changed during execution')
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');code=1
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        try:owner.stop()
        except BaseException as exc:report.update(status='failed',cleanup_error=str(exc));code=1
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        export_replay(out,destination)
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'work_directory':str(out),'export_directory':str(destination)}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
