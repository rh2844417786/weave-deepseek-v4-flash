#!/usr/bin/env python3
"""Same 32 natural requests on fresh native and P0 services; differences record-only."""
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
from scripts.check_sglang_service import wait_ready
from scripts.run_gpqa_c32 import baseline_coverage, validate_runtime, _safe
from src.weave.calibration import atomic_json
from src.weave.c32_comparison import digest
from src.weave.gpu_allocation import allocated_gpus
from src.weave.runtime_contract import load_contract
from src.weave.performance import flush_cache_when_idle
from src.weave.ops import kernel_source_sha256
from src.weave.native_p0_pair import compare_clients, paired_graph_check, normalize_launch, validate_prior_smoke
from src.weave.native_p0_evidence import native_checkpoint_proof
from src.weave.native_p0 import native_p0_source_sha256
from src.weave.graph_hooks import graph_observer_source_sha256


def client_command(image,work,records,relative,name,port,timeout):
    records=Path(records)
    return ['docker','run','--rm','--pull=never','--name','weave-'+name,'--network=host',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work',
        '-v',f'{records.parent}:/prepared:ro',image,load_contract()['python_executable'],
        '/workspace/scripts/bench_native_p0_pair.py','--records',f'/prepared/{records.name}',
        '--base-url',f'http://127.0.0.1:{port}','--output-dir',f'/work/{relative}',
        '--timeout',str(timeout),'--controller-handshake','--execute']


def load_deadline(marker,started,now,timeout,drain):
    begin,end=marker.get('started_monotonic'),marker.get('deadline_monotonic')
    if (any(type(v)not in (int,float) or not math.isfinite(v) for v in (begin,end))
            or not started<=begin<=now or not math.isclose(end-begin,timeout,rel_tol=0,abs_tol=1e-6)):
        raise ValueError('invalid paired client deadline marker')
    return end+drain+30.


def export_public(run,destination):
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh paired public destination required')
    names=['experiment.json','comparison.json','selection.json','serving-source-manifest.json',
           'native-checkpoint-proof.json','p0-smoke-reuse.json','launch-comparison.json']
    per_group=['group.json','launch.json','runtime.json','server-lifecycle.json','client-lifecycle.json',
               'paired-client-summary.json','paired-load-start.json','warmup-ready.json',
               'graph-before.json','graph-after.json','graph-check.json']
    names.extend(f'{group}/{name}' for group in ('native','p0') for name in per_group)
    files={}
    for name in names:
        path=run/name
        if path.is_symlink():raise ValueError('public paired symlink forbidden')
        if path.is_file():
            raw=path.read_bytes();_safe(json.loads(raw));files[name]=raw
    destination.mkdir(parents=True)
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o664)
    manifest=dict(schema_version=1,status=json.loads(files['experiment.json'])['status'],raw_content_exported=False,
                  files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    atomic_json(destination/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('work-root','records'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--native-baseline-dir',type=Path,default=ROOT/'reports/server/gpqa-c32-9f1635df2afe')
    parser.add_argument('--p0-evidence-dir',type=Path,default=ROOT/'reports/server/gpqa-c32-dcb642236285')
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--order',choices=('native,p0','p0,native'),default='native,p0')
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--warmup-timeout',type=float,default=960.)
    parser.add_argument('--request-timeout',type=float,default=3600.)
    parser.add_argument('--drain-timeout',type=float,default=60.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--graph-evidence-port-base',type=int,default=30310)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    for name in ('work_root','records','model_path','native_baseline_dir','p0_evidence_dir','export_root'):
        value=getattr(args,name)
        if value is not None:
            if not value.is_absolute():parser.error(name+' must be absolute')
            setattr(args,name,value.resolve())
    for name in ('setup_timeout','warmup_timeout','request_timeout','drain_timeout'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:parser.error('finite positive timeouts required')
    ports=[args.port]+list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4))
    if min(ports)<1 or max(ports)>65535 or len(set(ports))!=5:parser.error('distinct valid localhost ports required')
    order=args.order.split(',');name='native-p0-pair-'+uuid.uuid4().hex[:12]
    out=args.work_root/name;destination=(args.export_root or args.work_root/'public-reports')/name
    contract=load_contract()
    def server_command(arm,checkpoint):
        return service_command('native',args.work_root,args.model_path,'native-ep-'+name+'-'+arm,args.port,30110,checkpoint,
            decode_graph='full',graph_evidence_port_base=args.graph_evidence_port_base,native_p0=arm=='p0')
    if not args.execute:
        print(json.dumps(dict(status='preview_not_executed',order=order,request_count=32,
            selection_policy='first_32_prepared_file_order',completion_policy='natural_EOS_or_32768_per_request',
            formal_performance=False,numerical_policy='record_only',request_timeout=args.request_timeout,
            p0_configuration=dict(stride=128,capacity=64),no_refill=True,
            services={arm:server_command(arm,'0'*64) for arm in order},
            clients={arm:client_command(contract['image'],args.work_root,args.records,f'{name}/{arm}',name+'-client-'+arm,args.port,args.request_timeout) for arm in order}),indent=2));return 0
    from src.weave.gpqa_c32 import load_gpqa_workload
    from src.weave.native_p0_pair_client import selection_identity
    from src.weave.checkpoint_identity import checkpoint_stats
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False)
    owner,client_owner=OwnedChild(),OwnedChild();code=1
    report=dict(schema_version=1,status='running',scope='native_P0_paired_generation',numerical_policy='record_only',
        generation_comparison_policy='record_only',formal_performance=False,accuracy_evaluated=False,
        order=order,request_count=32,gpuq_job_id=os.environ.get('GPUQ_JOB_ID'),host_gpu_allocation=allocation.csv,
        started_utc=datetime.now(timezone.utc).isoformat(),stage='identity',groups={})
    def save():atomic_json(out/'experiment.json',report)
    def cancel(number,frame):
        for child in (owner,client_owner):child.requested=number
        if not any(child.protected for child in (owner,client_owner)):raise InterruptedError('paired_job_cancelled')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    def check_cancelled():owner.check_cancelled();client_owner.check_cancelled()
    def read(path):return json.loads(path.read_text())
    normalized_launch=None
    try:
        save();workload=load_gpqa_workload(args.records,model_path=args.model_path);selection=selection_identity(workload)
        atomic_json(out/'selection.json',selection)
        source=serving_source_manifest();source_digest=digest(source);atomic_json(out/'serving-source-manifest.json',source)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        checkpoint=native_checkpoint_proof(args.native_baseline_dir,args.model_path);atomic_json(out/'native-checkpoint-proof.json',checkpoint)
        identity=dict(image_id=image,source_sha256=kernel_source_sha256(),serving_source_sha256=source_digest,
            checkpoint_content_sha256=checkpoint['checkpoint_content_sha256'],native_p0_source_sha256=native_p0_source_sha256(),
            observer_source_sha256=graph_observer_source_sha256(),graph_sources=read(ROOT/'configs/sglang_graph_sources.json')['sha256'])
        smoke=validate_prior_smoke(args.p0_evidence_dir,identity);atomic_json(out/'p0-smoke-reuse.json',smoke)
        report.update(identity=identity,selection=selection);save()
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
            'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        for arm in order:
            check_cancelled();folder=out/arm;folder.mkdir();server_dir=args.work_root/('native-ep-'+name+'-'+arm)
            group=dict(status='running',arm=arm,identity=identity,p0_enabled=arm=='p0');report['groups'][arm]=group
            report.update(stage='serving_'+arm);save()
            try:
                assert_ports_free(ports)
                with (folder/'server-launch-private.log').open('w') as log:
                    server=owner.spawn(server_command(arm,identity['checkpoint_content_sha256']),server_dir/'container_lifecycle.json',env=env,stdout=log,stderr=subprocess.STDOUT)
                    url=f'http://127.0.0.1:{args.port}';wait_ready(url,timeout=args.setup_timeout,process=server)
                    validate_runtime(read(server_dir/'runtime.json'),identity,allocation.csv)
                    launch=read(server_dir/'launch.json')
                    if launch.get('backend')!='native' or launch.get('graph_mode')!='full' or launch.get('native_p0')is not (arm=='p0'):
                        raise ValueError('paired launch backend/P0/full-graph differs')
                    actual=normalize_launch(launch['docker_argv'])
                    if normalized_launch is None:normalized_launch=actual
                    elif actual!=normalized_launch:raise ValueError('paired service launch changed beyond P0 and owned paths')
                    atomic_json(out/'launch-comparison.json',dict(status='matched_so_far',normalized_launch_sha256=digest(actual),checked_arms=list(report['groups']),allowed_differences=['P0 env flags','owned container name','owned output directory']))
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    command=client_command(image,args.work_root,args.records,f'{name}/{arm}',name+'-client-'+arm,args.port,args.request_timeout)
                    lifecycle=folder/'client-lifecycle.json'
                    manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
                        '--evidence',str(lifecycle),'--log',str(folder/'client-private.log'),'--']+command
                    child=client_owner.spawn(manager,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
                    started=time.monotonic();deadline=started+args.warmup_timeout;baseline=False;marker=None
                    while child.poll() is None:
                        check_cancelled()
                        if server.poll()is not None:raise RuntimeError('paired_service_exited_during_client')
                        if not baseline and (folder/'warmup-ready.json').is_file():
                            ready=read(folder/'warmup-ready.json').get('ready_monotonic');now=time.monotonic()
                            if type(ready)not in (int,float) or not math.isfinite(ready) or ready>now:raise ValueError('invalid paired warmup marker')
                            until=min(now+20.,ready+20.)
                            before=baseline_coverage(args.graph_evidence_port_base,deadline=until,endpoint='graph-coverage')
                            atomic_json(folder/'graph-before.json',before)
                            if time.monotonic()>=until:raise TimeoutError('paired baseline deadline')
                            atomic_json(folder/'measurement-go.json',{'status':'baseline_captured'});baseline=True
                        if (folder/'paired-load-start.json').is_file():
                            current=read(folder/'paired-load-start.json')
                            if not baseline or marker is not None and marker!=current:raise ValueError('paired load marker changed or precedes baseline')
                            marker=current;deadline=load_deadline(marker,started,time.monotonic(),args.request_timeout,args.drain_timeout)
                        if time.monotonic()>=deadline:raise TimeoutError('paired client deadline')
                        time.sleep(.2)
                    if child.returncode!=0:raise RuntimeError('paired client failed; inspect sanitized client summary')
                    # A short natural-EOS cohort can finish between controller polls.
                    # Read its immutable start marker once more after normal exit.
                    if (folder/'paired-load-start.json').is_file():
                        final_marker=read(folder/'paired-load-start.json')
                        if marker is not None and final_marker!=marker:raise ValueError('paired load marker changed at exit')
                        load_deadline(final_marker,started,time.monotonic(),args.request_timeout,args.drain_timeout)
                        marker=final_marker
                    client_owner.stop();check_cancelled()
                    if not baseline or marker is None:raise ValueError('missing paired measurement handshake')
                    summary=read(folder/'paired-client-summary.json')
                    if summary.get('selection')!=selection or summary.get('source_workload_identity')!=workload.identity:
                        raise ValueError('paired client input identity differs from controller')
                    # Each arm is valid independently; content comparisons cannot fail it.
                    compare_clients(summary,summary)
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    after=baseline_coverage(args.graph_evidence_port_base,deadline=time.monotonic()+30.,endpoint='graph-coverage')
                    atomic_json(folder/'graph-after.json',after)
                    graph=paired_graph_check(before,after,p0_enabled=arm=='p0',identity=identity);atomic_json(folder/'graph-check.json',graph)
                    group.update(status='passed',graph_check=graph)
                owner.stop();check_cancelled()
            except BaseException as exc:
                group.update(status='failed',error_type=type(exc).__name__);raise
            finally:
                atomic_json(folder/'group.json',group)
                for src,dst in (('launch.json','launch.json'),('runtime.json','runtime.json'),('container_lifecycle.json','server-lifecycle.json')):
                    if (server_dir/src).is_file():atomic_json(folder/dst,read(server_dir/src))
                save()
        if digest(serving_source_manifest())!=source_digest:raise ValueError('source changed during paired run')
        if load_gpqa_workload(args.records,model_path=args.model_path).identity!=workload.identity:raise ValueError('prepared data changed during paired run')
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed during paired run')
        if validate_prior_smoke(args.p0_evidence_dir,identity)!=smoke:raise ValueError('prior smoke proof changed')
        launch_check=read(out/'launch-comparison.json');launch_check['status']='matched';atomic_json(out/'launch-comparison.json',launch_check)
        comparison=compare_clients(read(out/'native/paired-client-summary.json'),read(out/'p0/paired-client-summary.json'))
        comparison.update(execution_order=order,graph_evidence={arm:report['groups'][arm]['graph_check'] for arm in order})
        comparison['sampling_interpretation']=('P0 runtime sampling occurred; before/after counters do not locate each output divergence relative to the first sample.' if report['groups']['p0']['graph_check']['p0_runtime_sampled'] else
            'No P0 runtime sample occurred during this cohort. Any observed output difference therefore cannot be attributed to a runtime sample within this cohort; captured event nodes and retained buffers still differ.')
        atomic_json(out/'comparison.json',comparison)
        report.update(status='paired_check_complete',stage='complete');code=0
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        private=out/'controller-error-private.log';private.write_text(str(exc)+'\n');private.chmod(0o600)
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={label:pool.submit(child.stop) for label,child in (('server',owner),('client',client_owner))}
            for label,future in futures.items():
                try:future.result()
                except BaseException as exc:
                    report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
                    path=out/(label+'-cleanup-private.log');path.write_text(str(exc)+'\n');path.chmod(0o600)
        for arm in report['groups']:
            path=args.work_root/('native-ep-'+name+'-'+arm)/'container_lifecycle.json'
            if path.is_file():atomic_json(out/arm/'server-lifecycle.json',read(path))
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        try:export_public(out,destination)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(dict(status=report['status'],work_directory=str(out),export_directory=str(destination))),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
