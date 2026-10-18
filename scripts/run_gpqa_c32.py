#!/usr/bin/env python3
"""Two owned, offline GPQA Main C32 eager services, each measured for 1200 seconds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_c32_comparison import service_command, serving_source_manifest
from scripts.run_model_acceptance import OwnedChild, assert_ports_free
from scripts.check_sglang_service import wait_ready, coverage, http_json
from src.weave.calibration import atomic_json
from src.weave.c32_comparison import digest, scheduler_counts, validate_coverage_delta
from src.weave.experiment_export import ensure_public
from src.weave.gpu_allocation import allocated_gpus
from src.weave.ops import kernel_source_manifest
from src.weave.performance import flush_cache_when_idle
from src.weave.runtime_contract import load_contract

GROUPS=('native','hopper_prefill')
SECONDS=1200


def client_command(image,work,records,output_relative,name,port):
    records=Path(records)
    return ['docker','run','--rm','--pull=never','--name','weave-'+name,'--network=host',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work',
        '-v',f'{records.parent}:/prepared:ro',image,load_contract()['python_executable'],
        '/workspace/scripts/bench_gpqa_c32.py','--records',f'/prepared/{records.name}',
        '--base-url',f'http://127.0.0.1:{port}','--output-dir',f'/work/{output_relative}',
        '--controller-handshake','--execute']



def baseline_coverage(base, *, deadline):
    """Collect the idle baseline within the controller's shared handshake budget."""
    reports=[]
    for rank in range(4):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('idle_baseline_deadline')
        reports.append(http_json(f'http://127.0.0.1:{base+rank}/coverage',timeout=remaining))
        if time.monotonic()>=deadline:raise TimeoutError('idle_baseline_deadline')
    return reports


def measurement_deadline(marker,started,now,cleanup_seconds):
    begin,end=marker.get('started_monotonic'),marker.get('deadline_monotonic')
    if (type(begin) not in (float,int) or type(end) not in (float,int)
            or not math.isfinite(begin) or not math.isfinite(end)
            or not started<=begin<=now or not math.isclose(end-begin,SECONDS,rel_tol=0,abs_tol=1e-6)):
        raise ValueError('invalid_client_window_marker')
    return end+cleanup_seconds


def validate_runtime(runtime,identity,allocation):
    if (runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=identity['image_id']
            or runtime.get('kernel_source_sha256')!=identity['source_sha256']
            or runtime.get('host_gpu_allocation')!=allocation):
        raise ValueError('service_runtime_identity_differs')


def _safe(value):
    ensure_public(value)
    forbidden={'records','output_token_ids','rid','request_id','wire_rid','generated_text','completion','question','answer'}
    if isinstance(value,dict):
        if forbidden.intersection(value):raise ValueError('private_GPQA_payload_cannot_be_exported')
        for item in value.values():_safe(item)
    elif isinstance(value,list):
        for item in value:_safe(item)


def export_public(run,destination,*,proof_source=None,proof=None):
    """Static current-artifact allowlist plus independently verified original proof bytes."""
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh_public_destination_required')
    names=['experiment.json','comparison.json','source-manifest.json','serving-source-manifest.json',
           'data-manifest.json','prefill-proof-check.json']
    group_names=('group.json','launch.json','runtime.json','server-lifecycle.json','measurement-lifecycle.json',
                 'client-summary.json','measurement-summary.json','gpqa-rows.json','load-window.json',
                 'load-progress.json','warmup-ready.json','coverage-before.json','coverage-after.json',
                 'coverage-check.json','scheduler-summary.json')
    names += [f'{group}/{name}' for group in GROUPS for name in group_names]
    files={}
    for name in names:
        path=run/name
        if path.is_symlink():raise ValueError('public_symlink_forbidden')
        if path.is_file():
            raw=path.read_bytes();_safe(json.loads(raw));files[name]=raw
    destination.mkdir(parents=True)
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o664)
    if proof_source is not None and proof is not None:
        from src.weave.gpqa_c32 import copy_verified_prefill_bundle
        copy_verified_prefill_bundle(proof_source,destination/'prefill-proof',proof=proof)
        for path in sorted((destination/'prefill-proof').rglob('*')):
            if path.is_file():
                path.chmod(0o664);files[path.relative_to(destination).as_posix()]=path.read_bytes()
    manifest={'schema_version':1,'status':json.loads(files['experiment.json'])['status'],
              'raw_content_exported':False,'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}}
    atomic_json(destination/'manifest.json',manifest);(destination/'manifest.json').chmod(0o664)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('work-root','records','prefill-evidence-dir'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--warmup-timeout',type=float,default=960.)
    parser.add_argument('--drain-timeout',type=float,default=60.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    for key in ('work_root','records','prefill_evidence_dir','model_path','export_root'):
        path=getattr(args,key)
        if path is not None:
            if not path.is_absolute():parser.error(key+' must be absolute')
            setattr(args,key,path.resolve())
    if any(not math.isfinite(x) or x<=0 for x in (args.setup_timeout,args.warmup_timeout,args.drain_timeout)):
        parser.error('finite positive timeouts required')
    if not 1<=args.port<=65535 or not 1<=args.evidence_port_base<=65532 or args.port in range(args.evidence_port_base,args.evidence_port_base+4):
        parser.error('valid distinct serving/evidence ports required')
    name='gpqa-c32-'+uuid.uuid4().hex[:12]
    out=args.work_root/name;destination=(args.export_root or args.work_root/'public-reports')/name
    contract=load_contract()
    preview={v:service_command(v,args.work_root,args.model_path,'native-ep-'+name+'-'+v,args.port,args.evidence_port_base,'0'*64) for v in GROUPS}
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','groups':preview,'seconds':SECONDS,'concurrency':32,
            'max_new_tokens':32768,'ignore_eos':False,'thinking_mode':'thinking','reasoning_effort':'high',
            'numerical_policy':'record_only','rerun_gpu_matrix':False,'prefill_evidence_dir':str(args.prefill_evidence_dir),
            'required_environment':{'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'},
            'clients':{v:client_command(contract['image'],args.work_root,args.records,f'{name}/{v}',name+'-'+v,args.port) for v in GROUPS}},indent=2));return 0
    from src.weave.gpqa_c32 import load_gpqa_workload, validate_client_summary, compare_groups, validate_prefill_bundle
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False)
    owner,client_owner=OwnedChild(),OwnedChild()
    report={'schema_version':1,'status':'running','scope':'GPQA_Main_C32_eager','formal_gpqa':True,
        'accuracy_evaluated':False,'numerical_policy':'record_only','generation_comparison_policy':'record_only',
        'started_utc':datetime.now(timezone.utc).isoformat(),'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),
        'host_gpu_allocation':allocation.csv,'groups':{},'stage':'validating','rerun_gpu_matrix':False}
    proof=None;code=1
    def save():atomic_json(out/'experiment.json',report)
    def cancel(number,frame):
        for child in (owner,client_owner):child.requested=number
        if not any(child.protected for child in (owner,client_owner)):raise InterruptedError('job_cancelled')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    def check_cancelled():owner.check_cancelled();client_owner.check_cancelled()
    try:
        save();workload=load_gpqa_workload(args.records,model_path=args.model_path)
        source=serving_source_manifest();source_digest=digest(source)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        proof=validate_prefill_bundle(args.prefill_evidence_dir,args.model_path,image)
        identity={**proof['identity'],**workload.identity,'image_id':image,'serving_source_sha256':source_digest}
        report.update(identity=identity,core_gate=proof['core_gate']);save()
        atomic_json(out/'data-manifest.json',workload.identity)
        atomic_json(out/'source-manifest.json',kernel_source_manifest());atomic_json(out/'serving-source-manifest.json',source)
        atomic_json(out/'prefill-proof-check.json',proof)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
            'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        for variant in GROUPS:
            check_cancelled();report.update(stage='serving_'+variant);save()
            folder=out/variant;folder.mkdir();run_name='native-ep-'+name+'-'+variant
            server_dir=args.work_root/run_name;url=f'http://127.0.0.1:{args.port}'
            group={'status':'running','variant':variant,'stage':'starting','identity':identity}
            report['groups'][variant]=group
            try:
                assert_ports_free([args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4)))
                with (folder/'server-launch-private.log').open('w') as log:
                    server=owner.spawn(service_command(variant,args.work_root,args.model_path,run_name,args.port,args.evidence_port_base,identity['checkpoint_content_sha256']),server_dir/'container_lifecycle.json',env=env,stdout=log,stderr=subprocess.STDOUT)
                    wait_ready(url,timeout=args.setup_timeout,process=server)
                    runtime=json.loads((server_dir/'runtime.json').read_text());validate_runtime(runtime,identity,allocation.csv)
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    command=client_command(image,args.work_root,args.records,f'{name}/{variant}',name+'-'+variant,args.port)
                    lifecycle=folder/'measurement-lifecycle.json'
                    manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
                             '--evidence',str(lifecycle),'--log',str(folder/'client-private.log'),'--']+command
                    child=client_owner.spawn(manager,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
                    started=time.monotonic();deadline=started+args.warmup_timeout;marker=None;baseline=False
                    while child.poll() is None:
                        check_cancelled();now=time.monotonic()
                        if server.poll() is not None:raise RuntimeError('service_exited_during_client')
                        if not baseline and (folder/'warmup-ready.json').is_file():
                            ready=json.loads((folder/'warmup-ready.json').read_text())
                            ready_time=ready.get('ready_monotonic')
                            baseline_now=time.monotonic()
                            if type(ready_time) not in (float,int) or not math.isfinite(ready_time) or ready_time>baseline_now:
                                raise ValueError('invalid_warmup_ready_marker')
                            baseline_deadline=min(baseline_now+20.,ready_time+20.)
                            before=baseline_coverage(args.evidence_port_base,deadline=baseline_deadline) if variant!='native' else None
                            if before is not None:atomic_json(folder/'coverage-before.json',before)
                            log_offset=(server_dir/'server.log').stat().st_size
                            if time.monotonic()>=baseline_deadline:raise TimeoutError('idle_baseline_deadline')
                            atomic_json(folder/'measurement-go.json',{'status':'baseline_captured'})
                            baseline=True;group['stage']='measurement';atomic_json(folder/'group.json',group)
                        if (folder/'load-window.json').is_file():
                            current=json.loads((folder/'load-window.json').read_text())
                            if not baseline:raise ValueError('measurement_started_before_idle_baseline')
                            if marker is not None and current!=marker:raise ValueError('client_window_marker_changed')
                            marker=current;deadline=measurement_deadline(marker,started,time.monotonic(),args.drain_timeout+30)
                        if time.monotonic()>=deadline:raise TimeoutError('client_measurement_deadline' if marker else 'client_warmup_deadline')
                        time.sleep(.2)
                    if child.returncode!=0:raise RuntimeError('measurement_client_failed')
                    client_owner.stop();check_cancelled()
                    if marker is None or not baseline:raise ValueError('missing_measurement_window')
                    client=json.loads((folder/'measurement-summary.json').read_text());validate_client_summary(client)
                    if client.get('workload')!=workload.identity:raise ValueError('client_workload_identity_differs')
                    atomic_json(folder/'client-summary.json',client)
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    with (server_dir/'server.log').open('rb') as stream:stream.seek(log_offset);logs=stream.read().decode(errors='replace')
                    scheduler=scheduler_counts(logs);atomic_json(folder/'scheduler-summary.json',scheduler)
                    if variant=='native':
                        if scheduler['max_running_requests']!=32:raise ValueError('native_actual_concurrency32_not_observed')
                        batch=scheduler
                    else:
                        after=coverage(args.evidence_port_base);atomic_json(folder/'coverage-after.json',after)
                        keys=('candidate_source_sha256','assembly_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256')
                        batch=validate_coverage_delta(before,after,'hopper_fp8',identity['source_sha256'],identity['checkpoint_content_sha256'],prefill_implementation='hopper_fp8',**{key:identity[key] for key in keys})
                        atomic_json(folder/'coverage-check.json',batch)
                    group.update(status='passed',stage='complete',client_summary=client,batch_evidence=batch)
                owner.stop();check_cancelled()
            except BaseException as exc:
                group.update(status='failed',error_type=type(exc).__name__)
                raise
            finally:
                atomic_json(folder/'group.json',group)
                for src,dst in (('launch.json','launch.json'),('runtime.json','runtime.json'),('container_lifecycle.json','server-lifecycle.json')):
                    if (server_dir/src).is_file():atomic_json(folder/dst,json.loads((server_dir/src).read_text()))
                save()
        if load_gpqa_workload(args.records,model_path=args.model_path).identity!=workload.identity:raise ValueError('data_changed_during_run')
        if digest(serving_source_manifest())!=source_digest:raise ValueError('source_changed_during_run')
        if validate_prefill_bundle(args.prefill_evidence_dir,args.model_path,image)!=proof:raise ValueError('proof_or_checkpoint_changed_during_run')
        atomic_json(out/'comparison.json',compare_groups(report['groups']))
        report.update(status='comparison_complete',stage='complete');code=0
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        (out/'controller-error-private.log').write_text(str(exc)+'\n')
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={label:pool.submit(child.stop) for label,child in (('server',owner),('client',client_owner))}
            for label,future in futures.items():
                try:future.result()
                except BaseException as exc:
                    report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
                    (out/(label+'-cleanup-error-private.log')).write_text(str(exc)+'\n')
        for variant in report['groups']:
            lifecycle=args.work_root/('native-ep-'+name+'-'+variant)/'container_lifecycle.json'
            if lifecycle.is_file():atomic_json(out/variant/'server-lifecycle.json',json.loads(lifecycle.read_text()))
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        try:export_public(out,destination,proof_source=args.prefill_evidence_dir,proof=proof)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'work_directory':str(out),'export_directory':str(destination)}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
