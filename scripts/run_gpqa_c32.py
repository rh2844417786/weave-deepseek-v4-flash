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
from src.weave.graph_evidence import validate_graph_delta, validate_graph_weave_delta, validate_graph_smoke
from scripts.run_graph_gate import graph_gate_command, stage_event_gate_command
from src.weave.graph_stage_evidence import summarize_graph_stages, validate_stage_event_smoke

GROUPS=('native','hopper_prefill')
PAPER_GROUPS=('native','paper_persistent')
SECONDS=1200


def paper_service_command(variant,work,model,name,port,base,checkpoint,*,paper_profile_dir=None,**options):
    mapped='hopper_prefill' if variant=='paper_persistent' else variant
    command=service_command(mapped,work,model,name,port,base,checkpoint,**options)
    if variant=='paper_persistent':
        if paper_profile_dir is None:raise ValueError('paper profile required')
        for flag in ('--decode-implementation','--prefill-implementation'):
            command[command.index(flag)+1]='hopper_persistent'
        command+=['--profile',str(paper_profile_dir)]
    return command


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



def wait_graph_gate(child, folder, *, gate_timeout, cleanup_timeout, cancel_check=lambda: None):
    """Bound teardown from the parent, including a worker blocked inside C++."""
    deadline = time.monotonic() + gate_timeout
    cleanup_deadline = None
    while child.poll() is None:
        cancel_check()
        for rank in range(4):
            path = Path(folder) / f'graph-cleanup-rank{rank}.json'
            if not path.exists():
                continue
            state = json.loads(path.read_text())
            started = state.get('started_monotonic')
            if type(started) not in (int, float) or not math.isfinite(started) or not 0 <= started <= time.monotonic():
                raise ValueError('invalid_graph_cleanup_clock')
            candidate = started + cleanup_timeout
            cleanup_deadline = candidate if cleanup_deadline is None else min(cleanup_deadline, candidate)
            if state.get('status') == 'failed':
                raise RuntimeError('graph_cleanup_failed')
        now = time.monotonic()
        if cleanup_deadline is not None and now >= cleanup_deadline:
            raise TimeoutError('graph_cleanup_deadline')
        if now >= deadline:
            raise TimeoutError('graph_gpu_gate_deadline')
        time.sleep(.2)
    if child.returncode != 0:
        raise RuntimeError('graph_gpu_gate_failed')


def baseline_coverage(base, *, deadline, endpoint="coverage"):
    """Collect the idle baseline within the controller's shared handshake budget."""
    reports=[]
    for rank in range(4):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('idle_baseline_deadline')
        reports.append(http_json(f'http://127.0.0.1:{base+rank}/{endpoint}',timeout=remaining))
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


def export_public(run,destination,*,proof_source=None,proof=None,paper_source=None,paper_proof=None):
    """Static current-artifact allowlist plus independently verified original proof bytes."""
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh_public_destination_required')
    names=['experiment.json','comparison.json','source-manifest.json','serving-source-manifest.json',
           'data-manifest.json','prefill-proof-check.json','graph-proof-check.json','graph-gate/runtime.json',
           'graph-gate/container-lifecycle.json','paper-proof-check.json','native-checkpoint-proof.json','native-p0-summary.json','native-p0-proof-check.json','native-p0-gate/runtime.json','native-p0-gate/container-lifecycle.json']
    names += [f'native-p0-gate/{prefix}-rank{rank}.json' for prefix in ('native-p0-event','graph-cleanup') for rank in range(4)]
    names += [f'graph-gate/hopper-graph-rank{rank}.json' for rank in range(4)]
    names += [f'graph-gate/graph-cleanup-rank{rank}.json' for rank in range(4)]
    names += ['stage-event-proof-check.json','stage-event-gate/runtime.json','stage-event-gate/container-lifecycle.json','stage-diagnostic-summary.json']
    names += [f'stage-event-gate/{prefix}-rank{rank}.json' for prefix in ('stage-event','graph-cleanup') for rank in range(4)]
    group_names=('group.json','launch.json','runtime.json','server-lifecycle.json','measurement-lifecycle.json',
                 'client-summary.json','measurement-summary.json','gpqa-rows.json','load-window.json',
                 'load-progress.json','warmup-ready.json','coverage-before.json','coverage-after.json',
                 'coverage-check.json','scheduler-summary.json','graph-before.json','graph-after.json','graph-check.json','stage-profile-check.json')
    names += [f'{group}/{name}' for group in (*GROUPS,'paper_persistent') for name in group_names]
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
    if paper_source is not None and paper_proof is not None:
        for name,expected in paper_proof['files'].items():
            source=Path(paper_source)/name
            if source.is_symlink():raise ValueError('paper proof symlink forbidden')
            raw=source.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('paper proof changed before export')
            _safe(json.loads(raw));target=destination/'paper-proof'/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o664)
            files['paper-proof/'+name]=raw
    manifest={'schema_version':1,'status':json.loads(files['experiment.json'])['status'],
              'raw_content_exported':False,'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}}
    atomic_json(destination/'manifest.json',manifest);(destination/'manifest.json').chmod(0o664)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('work-root','records'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--prefill-evidence-dir',type=Path)
    parser.add_argument('--native-p0',action='store_true')
    parser.add_argument('--native-p0-stride',type=int,default=128)
    parser.add_argument('--native-p0-capacity',type=int,default=64)
    parser.add_argument('--native-baseline-dir',type=Path,default=ROOT/'reports/server/gpqa-c32-9f1635df2afe')
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--warmup-timeout',type=float,default=960.)
    parser.add_argument('--drain-timeout',type=float,default=60.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--decode-graph',choices=('disabled','full'),default='disabled')
    parser.add_argument('--graph-evidence-port-base',type=int,default=30310)
    parser.add_argument('--graph-gate-timeout',type=float,default=7200.)
    parser.add_argument('--graph-cleanup-timeout',type=float,default=120.)
    parser.add_argument('--paper-proof-dir',type=Path)
    parser.add_argument('--graph-stage-profile',action='store_true')
    parser.add_argument('--graph-stage-stride',type=int,default=128)
    parser.add_argument('--graph-stage-capacity',type=int,default=64)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    native_p0=args.native_p0
    if native_p0 and (args.decode_graph!='full' or args.graph_stage_profile or args.paper_proof_dir or args.prefill_evidence_dir):parser.error('native P0 requires native full graph without candidate proofs/profilers')
    if not native_p0 and args.prefill_evidence_dir is None:parser.error('--prefill-evidence-dir required outside native P0')
    if not 1<=args.native_p0_stride<=65536 or not 1<=args.native_p0_capacity<=256:parser.error('invalid native P0 sampling bounds')
    paper_mode=args.paper_proof_dir is not None
    if paper_mode and (args.decode_graph!='full' or args.graph_stage_profile):parser.error('paper verification requires full graph without operator stage profiler')
    groups=('native',) if native_p0 else PAPER_GROUPS if paper_mode else GROUPS
    if args.graph_stage_profile and args.decode_graph!='full':parser.error('stage diagnostic requires full graph')
    if not 1<=args.graph_stage_stride<=65536 or not 1<=args.graph_stage_capacity<=256:parser.error('invalid stage sampling bounds')
    measurement_mode='instrumented_diagnostic' if (args.graph_stage_profile or native_p0) else 'uninstrumented_performance'
    for key in ('work_root','records','prefill_evidence_dir','model_path','export_root','paper_proof_dir','native_baseline_dir'):
        path=getattr(args,key)
        if path is not None:
            if not path.is_absolute():parser.error(key+' must be absolute')
            setattr(args,key,path.resolve())
    if any(not math.isfinite(x) or x<=0 for x in (args.setup_timeout,args.warmup_timeout,args.drain_timeout,args.graph_gate_timeout,args.graph_cleanup_timeout)):
        parser.error('finite positive timeouts required')
    if not 1<=args.port<=65535 or not 1<=args.evidence_port_base<=65532 or args.port in range(args.evidence_port_base,args.evidence_port_base+4):
        parser.error('valid distinct serving/evidence ports required')
    graph_options={'decode_graph':args.decode_graph,'graph_evidence_port_base':args.graph_evidence_port_base}
    service_options=dict(paper_profile_dir=args.paper_proof_dir/'calibrate') if paper_mode else {}
    if args.graph_stage_profile:graph_options.update(graph_stage_profile=True,graph_stage_stride=args.graph_stage_stride,graph_stage_capacity=args.graph_stage_capacity)
    if native_p0:graph_options.update(native_p0=True,native_p0_stride=args.native_p0_stride,native_p0_capacity=args.native_p0_capacity)
    ports=[args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4))
    if args.decode_graph=='full':
        ports+=list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4))
        if min(ports)<1 or max(ports)>65535 or len(set(ports))!=len(ports):parser.error('valid distinct graph ports required')
    name='gpqa-c32-'+uuid.uuid4().hex[:12]
    out=args.work_root/name;destination=(args.export_root or args.work_root/'public-reports')/name
    contract=load_contract()
    preview={v:paper_service_command(v,args.work_root,args.model_path,'native-ep-'+name+'-'+v,args.port,args.evidence_port_base,'0'*64,**graph_options,**service_options) for v in groups}
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','groups':preview,'seconds':SECONDS,'concurrency':32,
            'max_new_tokens':32768,'ignore_eos':False,'thinking_mode':'thinking','reasoning_effort':'high',
            'decode_graph':args.decode_graph,'graph_smoke_required':args.decode_graph=='full' and not paper_mode and not native_p0,
            'graph_gate_required':args.decode_graph=='full' and not paper_mode and not native_p0,
            'graph_cleanup_timeout':args.graph_cleanup_timeout,
            'measurement_mode':measurement_mode,'formal_gpqa':not (args.graph_stage_profile or native_p0),
            'native_p0_gate_required':native_p0,'stage_event_gate_required':args.graph_stage_profile,'paper_mode':paper_mode,'complete_weave_claim':False,
            'numerical_policy':'record_only','rerun_gpu_matrix':False,'prefill_evidence_dir':str(args.prefill_evidence_dir) if args.prefill_evidence_dir else None,'native_baseline_dir':str(args.native_baseline_dir) if native_p0 else None,
            'required_environment':{'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'},
            'clients':{v:client_command(contract['image'],args.work_root,args.records,f'{name}/{v}',name+'-'+v,args.port) for v in groups}},indent=2));return 0
    from src.weave.gpqa_c32 import load_gpqa_workload, validate_client_summary, compare_groups, validate_prefill_bundle
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False)
    owner,client_owner=OwnedChild(),OwnedChild()
    report={'schema_version':1,'status':'running','scope':'native_sglang_p0_c32_full' if native_p0 else 'GPQA_Main_C32_'+args.decode_graph,'formal_gpqa':not (args.graph_stage_profile or native_p0),
        'measurement_mode':measurement_mode,
        'accuracy_evaluated':False,'numerical_policy':'record_only','generation_comparison_policy':'record_only',
        'started_utc':datetime.now(timezone.utc).isoformat(),'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),
        'host_gpu_allocation':allocation.csv,'groups':{},'stage':'validating','rerun_gpu_matrix':False}
    proof=None;paper_proof=None;code=1
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
        if native_p0:
            from src.weave.native_p0_evidence import native_checkpoint_proof, summarize_native_p0, validate_native_p0_smoke
            from src.weave.native_p0 import native_p0_source_sha256
            from scripts.run_graph_gate import native_p0_gate_command
            native_proof=native_checkpoint_proof(args.native_baseline_dir,args.model_path)
            atomic_json(out/'native-checkpoint-proof.json',native_proof)
            proof_identity={'source_sha256':digest(kernel_source_manifest()),'checkpoint_content_sha256':native_proof['checkpoint_content_sha256']}
        else:
            proof=validate_prefill_bundle(args.prefill_evidence_dir,args.model_path,image)
            proof_identity=proof['identity']
        identity={**proof_identity,**workload.identity,'image_id':image,'serving_source_sha256':source_digest,'decode_graph':args.decode_graph}
        if args.decode_graph=='full':
            from src.weave.hopper_graph import graph_runtime_source_sha256
            from src.weave.graph_hooks import graph_observer_source_sha256
            identity.update(graph_runtime_source_sha256=graph_runtime_source_sha256(),observer_source_sha256=graph_observer_source_sha256(),graph_sources=json.loads((ROOT/'configs/sglang_graph_sources.json').read_text())['sha256'])
        identity['measurement_mode']=measurement_mode
        if args.graph_stage_profile:
            from src.weave.graph_stage_profile import graph_stage_source_sha256
            identity.update(graph_stage_source_sha256=graph_stage_source_sha256(),graph_stage_stride=args.graph_stage_stride,graph_stage_capacity=args.graph_stage_capacity)
        if native_p0:identity.update(native_p0_source_sha256=native_p0_source_sha256(),native_p0_stride=args.native_p0_stride,native_p0_capacity=args.native_p0_capacity)
        if paper_mode:
            from src.weave.paper_evidence import validate_paper_bundle
            paper_proof=validate_paper_bundle(args.paper_proof_dir,identity)
            identity.update(paper_source_sha256=paper_proof['paper_source_sha256'],paper_profiles=paper_proof['profile_sha256'])
            report.update(method_status='paper_candidate',complete_weave_claim=False)
            atomic_json(out/'paper-proof-check.json',paper_proof)
        report.update(identity=identity,core_gate=proof['core_gate'] if proof else 'not_applicable_native_only');save()
        atomic_json(out/'data-manifest.json',workload.identity)
        atomic_json(out/'source-manifest.json',kernel_source_manifest());atomic_json(out/'serving-source-manifest.json',source)
        if proof is not None:atomic_json(out/'prefill-proof-check.json',proof)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
            'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        if args.decode_graph=='full' and not paper_mode and not native_p0:
            gate_dir=out/'graph-gate';gate_dir.mkdir()
            gate=graph_gate_command(image,args.work_root,args.model_path,name,allocation.csv)
            lifecycle=gate_dir/'container-lifecycle.json'
            manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',gate[gate.index('--name')+1],
                     '--evidence',str(lifecycle),'--log',str(gate_dir/'gate-private.log'),'--']+gate
            child=owner.spawn(manager,lifecycle,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            report.update(stage='graph_gate',graph_cleanup_timeout=args.graph_cleanup_timeout);save()
            wait_graph_gate(child,gate_dir,gate_timeout=args.graph_gate_timeout,
                            cleanup_timeout=args.graph_cleanup_timeout,cancel_check=check_cancelled)
            owner.stop();check_cancelled()
            gate_runtime=json.loads((gate_dir/'runtime.json').read_text());validate_runtime(gate_runtime,identity,allocation.csv)
            graph_proof=validate_graph_smoke(gate_dir,identity)
            atomic_json(out/'graph-proof-check.json',graph_proof)
        if args.graph_stage_profile:
            stage_dir=out/'stage-event-gate';stage_dir.mkdir()
            command=stage_event_gate_command(image,args.work_root,args.model_path,name,allocation.csv)
            lifecycle=stage_dir/'container-lifecycle.json'
            manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
                     '--evidence',str(lifecycle),'--log',str(stage_dir/'gate-private.log'),'--']+command
            report.update(stage='stage_event_gate');save()
            child=owner.spawn(manager,lifecycle,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            wait_graph_gate(child,stage_dir,gate_timeout=args.graph_gate_timeout,
                            cleanup_timeout=args.graph_cleanup_timeout,cancel_check=check_cancelled)
            owner.stop();check_cancelled()
            validate_runtime(json.loads((stage_dir/'runtime.json').read_text()),identity,allocation.csv)
            stage_proof=validate_stage_event_smoke(stage_dir,identity)
            atomic_json(out/'stage-event-proof-check.json',stage_proof)
        if native_p0:
            gate_dir=out/'native-p0-gate';gate_dir.mkdir()
            command=native_p0_gate_command(image,args.work_root,args.model_path,name,allocation.csv)
            lifecycle=gate_dir/'container-lifecycle.json'
            manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
                     '--evidence',str(lifecycle),'--log',str(gate_dir/'gate-private.log'),'--']+command
            report.update(stage='native_p0_event_gate');save()
            child=owner.spawn(manager,lifecycle,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            wait_graph_gate(child,gate_dir,gate_timeout=args.graph_gate_timeout,
                            cleanup_timeout=args.graph_cleanup_timeout,cancel_check=check_cancelled)
            owner.stop();check_cancelled()
            validate_runtime(json.loads((gate_dir/'runtime.json').read_text()),identity,allocation.csv)
            native_smoke=validate_native_p0_smoke(gate_dir,identity)
            atomic_json(out/'native-p0-proof-check.json',native_smoke)
        for variant in groups:
            check_cancelled();report.update(stage='serving_'+variant);save()
            folder=out/variant;folder.mkdir();run_name='native-ep-'+name+'-'+variant
            server_dir=args.work_root/run_name;url=f'http://127.0.0.1:{args.port}'
            group={'status':'running','variant':variant,'stage':'starting','identity':identity}
            report['groups'][variant]=group
            try:
                assert_ports_free(ports)
                with (folder/'server-launch-private.log').open('w') as log:
                    server=owner.spawn(paper_service_command(variant,args.work_root,args.model_path,run_name,args.port,args.evidence_port_base,identity['checkpoint_content_sha256'],**graph_options,**service_options),server_dir/'container_lifecycle.json',env=env,stdout=log,stderr=subprocess.STDOUT)
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
                            if args.decode_graph=='full':
                                graph_before=baseline_coverage(args.graph_evidence_port_base,deadline=baseline_deadline,endpoint='graph-coverage')
                                atomic_json(folder/'graph-before.json',graph_before)
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
                    if args.decode_graph=='full':
                        graph_after=baseline_coverage(args.graph_evidence_port_base,deadline=time.monotonic()+20.,endpoint='graph-coverage')
                        atomic_json(folder/'graph-after.json',graph_after)
                        graph_check=validate_graph_delta(graph_before,graph_after,expected_observer_source=identity['observer_source_sha256'],expected_sources=identity['graph_sources'])
                        atomic_json(folder/'graph-check.json',graph_check)
                        if args.graph_stage_profile:
                            stages=summarize_graph_stages(graph_before,graph_after,
                                kind='native' if variant=='native' else 'weave',
                                profile_source=identity['graph_stage_source_sha256'],
                                stride=args.graph_stage_stride,capacity=args.graph_stage_capacity)
                            atomic_json(folder/'stage-profile-check.json',stages)
                            group['stage_profile']=stages
                        if native_p0:
                            native_summary=summarize_native_p0(graph_before,graph_after,profile_source=identity['native_p0_source_sha256'],stride=args.native_p0_stride,capacity=args.native_p0_capacity)
                            atomic_json(out/'native-p0-summary.json',native_summary)
                            group['native_p0_summary']=native_summary
                        if variant=='native':batch=graph_check
                        else:
                            after=coverage(args.evidence_port_base);atomic_json(folder/'coverage-after.json',after)
                            if paper_mode:
                                from src.weave.paper_evidence import validate_paper_coverage
                                batch=validate_paper_coverage(before,after,graph_check,identity,paper_proof)
                            else:batch=validate_graph_weave_delta(before,after,graph_check,identity)
                            atomic_json(folder/'coverage-check.json',batch)
                    elif variant=='native':
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
        if native_p0:
            from src.weave.checkpoint_identity import checkpoint_stats
            if checkpoint_stats(args.model_path)!=native_proof['verified_stats']:raise ValueError('checkpoint changed during native P0')
            if native_p0_source_sha256()!=identity['native_p0_source_sha256']:raise ValueError('native P0 source changed')
            if validate_native_p0_smoke(out/'native-p0-gate',identity)!=native_smoke:raise ValueError('native P0 smoke changed')
        elif validate_prefill_bundle(args.prefill_evidence_dir,args.model_path,image)!=proof:raise ValueError('proof_or_checkpoint_changed_during_run')
        if args.decode_graph=='full' and not paper_mode and not native_p0 and validate_graph_smoke(out/'graph-gate',identity)!=graph_proof:raise ValueError('graph_proof_changed_during_run')
        comparison_groups=report['groups'] if not paper_mode else {'native':report['groups']['native'],'hopper_prefill':report['groups']['paper_persistent']}
        comparison=({'status':'diagnostic_complete','formal_gpqa':False,'measurement_mode':measurement_mode,
            'decode_graph':'full','group_labels':{'native':'SGLang native with P0 instrumentation'},
            'sustained_c32_ratio_available':False,'ratio_suppression_reason':'native_only_instrumented_diagnostic'} if native_p0
            else dict(compare_groups(comparison_groups),decode_graph=args.decode_graph,measurement_mode=measurement_mode))
        if paper_mode:
            if validate_paper_bundle(args.paper_proof_dir,identity)!=paper_proof:raise ValueError('paper proof changed')
            comparison['decode_tokens_per_second']['paper_persistent']=comparison['decode_tokens_per_second'].pop('hopper_prefill')
            comparison['paper_vs_native_ratio']=comparison.pop('prefill_vs_native_ratio')
            comparison.update(method_identity='hopper_persistent_paper_v1',method_status='paper_service_execution_verified',complete_weave_claim=False)
        elif not native_p0:comparison['group_labels']={'native':'SGLang native','hopper_prefill':'Hopper operator candidate; paper scheduler not used'}
        if args.graph_stage_profile:
            if validate_stage_event_smoke(out/'stage-event-gate',identity)!=stage_proof:raise ValueError('stage_event_proof_changed')
            comparison.update(formal_gpqa=False,sustained_c32_ratio_available=False,prefill_vs_native_ratio=None,
                              ratio_suppression_reason='instrumented_stage_diagnostic')
            atomic_json(out/'stage-diagnostic-summary.json',{
                'status':'diagnostic_complete','measurement_mode':measurement_mode,'formal_performance':False,
                'numerical_policy':'record_only','identity':identity,
                'groups':{v:report['groups'][v]['stage_profile'] for v in GROUPS},
                'interpretation':'Use aligned stage times to locate the bottleneck; instrumented throughput is not a formal speedup comparison.'})
        atomic_json(out/'comparison.json',comparison)
        report.update(status='diagnostic_complete' if (args.graph_stage_profile or native_p0) else 'comparison_complete',stage='complete');code=0
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
        try:export_public(out,destination,proof_source=args.prefill_evidence_dir,proof=proof,paper_source=args.paper_proof_dir,paper_proof=paper_proof)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'work_directory':str(out),'export_directory':str(destination)}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
