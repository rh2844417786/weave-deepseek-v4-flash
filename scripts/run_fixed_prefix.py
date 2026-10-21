#!/usr/bin/env python3
"""Offline native repeat/P0 decomposition with identical short teacher prefixes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_model_acceptance import OwnedChild,assert_ports_free
from scripts.run_c32_comparison import serving_source_manifest
from scripts.run_gpqa_c32 import baseline_coverage,validate_runtime,wait_graph_gate
from scripts.run_graph_gate import native_p0_gate_command
from scripts.check_sglang_service import http_json,wait_ready
from src.weave.calibration import atomic_json
from src.weave.experiment_export import ensure_public
from src.weave.c32_serving import _count_and_terminal,_validate_response_identity
from src.weave.gpu_allocation import allocated_gpus
from src.weave.runtime_contract import load_contract
from src.weave.performance import flush_cache_when_idle
from src.weave.ops import kernel_source_sha256
from src.weave.graph_hooks import graph_observer_source_sha256
from src.weave.native_p0 import native_p0_source_sha256
from src.weave.native_p0_evidence import native_checkpoint_proof,validate_native_p0_smoke
from src.weave.fixed_prefix_data import prepare_config,validate_pinned_selection,load_teacher,digest
from src.weave.fixed_prefix_evidence import ARMS,normalize_fixed_launch,validate_fixed_graph,validate_phase_proofs
from src.weave.fixed_prefix_runtime import fixed_prefix_source_sha256
from src.weave.fixed_prefix_budget import FIXED_PAGE_SIZE,FIXED_PREFILL_TOKENS,admission_budget,validate_effective_budget,SERVER_FIELDS,effective_request_limits


def service_command(arm,work,model,name,port,base,checkpoint,private,teacher):
    command=[sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--execute','--backend','native',
        '--graph-c32','--decode-graph','full','--request-timing','--work-root',str(work),'--model-path',str(model),
        '--run-name','native-ep-'+name+'-'+arm,'--port',str(port),'--graph-evidence-port-base',str(base),
        '--checkpoint-content-sha256',checkpoint,'--fixed-prefix-dir',str(private),'--fixed-prefix-teacher',str(teacher)]
    if arm.startswith('p0_'):command+=['--native-p0','--native-p0-mode',arm.split('_',1)[1],'--native-p0-no-readback']
    return command


def cpu_container(image,work,name,*,network='none'):
    return ['docker','run','--rm','--pull=never','--name','weave-'+name,'--network='+network,
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work']


def phase_command(image,work,records,relative,teacher_relative,name,phase,port,timeout):
    records=Path(records)
    return cpu_container(image,work,name,network='host')+['-v',f'{records.parent}:/prepared:ro',image,load_contract()['python_executable'],
        '/workspace/scripts/bench_fixed_prefix.py','--records',f'/prepared/{records.name}','--base-url',f'http://127.0.0.1:{port}',
        '--output-dir',f'/work/{relative}','--teacher',f'/work/{teacher_relative}','--phase',phase,
        '--steps','2' if phase=='trace' else '8','--timeout',str(timeout),'--execute']


def _read(path):return json.loads(Path(path).read_text())


def warmup(url,workload,timeout):
    rids=[uuid.uuid4().hex for _ in range(32)]
    payload={'input_ids':[r['input_token_ids'] for r in workload.records[:32]],'rid':rids,
        'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':8},'stream':False,'return_logprob':False}
    response=http_json(url+'/generate',payload,timeout=timeout)
    if not isinstance(response,list) or len(response)!=32:raise ValueError('complete batched warmup required')
    by_id={r.get('meta_info',{}).get('id'):r for r in response}
    if set(by_id)!=set(rids):raise ValueError('warmup identities differ')
    for rid in rids:
        _validate_response_identity(by_id[rid],rid)
        count,terminal=_count_and_terminal(by_id[rid],8)
        if count!=8 or not terminal:raise ValueError('warmup actual token count incomplete')
    return {'status':'passed','requests_completed':32,'output_tokens':256,'included_in_measurement':False}


def check_prefill_budget(url,config,output):
    # The pinned /server_info response exposes resolved arguments at the top
    # level. Only the allowlisted budget fields are retained from that response.
    current=http_json(url+'/server_info',timeout=30)
    try:
        proof=validate_effective_budget([r['input_tokens'] for r in config['requests']],current)
    except ValueError:
        effective={}
        if isinstance(current,dict):
            for key in SERVER_FIELDS:
                if key in current:
                    value=current[key]
                    effective[key]=value if value is None or type(value)is int else {'invalid_type':type(value).__name__}
        try:worker_limits=effective_request_limits(current)
        except ValueError:worker_limits=None
        atomic_json(output,dict(status='failed',configuration_source='live_server_info',
            effective_max_running_requests_per_dp=worker_limits,
            error_code='effective_prefill_budget_invalid',effective_limits=effective,
            missing_fields=[key for key in SERVER_FIELDS if key not in effective]))
        raise
    atomic_json(output,proof)
    return proof


def public_scalars(value):
    # Fixed-prefix proofs call their scalar/hash step list "records". That name
    # is permitted here; raw benchmark fields remain forbidden at every depth.
    ensure_public(value)
    if isinstance(value,dict):
        if {'rid','request_id','wire_rid','generated_text','completion','question','answer','output_ids','input_ids'}.intersection(value):
            raise ValueError('private fixed-prefix content cannot be exported')
        for item in value.values():public_scalars(item)
    elif isinstance(value,list):
        for item in value:public_scalars(item)


def wait_phase_proofs(private_dir,*,phase,steps,config,teacher,identity,timeout,server,cancel_check):
    deadline=time.monotonic()+timeout
    while True:
        cancel_check()
        states=[]
        for rank in range(4):
            path=Path(private_dir)/f'{phase}-rank{rank}.json'
            states.append(_read(path).get('status') if path.is_file() else None)
        if 'failed' in states:raise RuntimeError('fixed-prefix rank phase failed')
        if states==['passed']*4:
            return validate_phase_proofs(private_dir,phase=phase,steps=steps,config=config,teacher=teacher,identity=identity)
        if server.poll() is not None:raise RuntimeError('service exited before all rank proofs completed')
        if time.monotonic()>=deadline:raise TimeoutError('fixed-prefix rank proof deadline')
        time.sleep(.1)


def export_public(run,destination):
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh public fixed-prefix destination required')
    names=['experiment.json','comparison.json','config.json','teacher-manifest.json','source-manifest.json',
        'checkpoint-proof.json','smoke-proof.json','launch-comparison.json','comparison-lifecycle.json','prefill-budget.json',
        'native-p0-gate/runtime.json','native-p0-gate/container-lifecycle.json']
    names += [f'native-p0-gate/{prefix}-rank{rank}.json' for prefix in ('native-p0-event','graph-cleanup') for rank in range(4)]
    for arm in ARMS:
        names += [f'{arm}/{name}' for name in ('group.json','launch.json','runtime.json','server-lifecycle.json','warmup.json','prefill-budget.json')]
        for phase in (('seed','measure','trace') if arm=='native_a' else ('measure',)):
            names += [f'{arm}/{phase}/{name}' for name in ('fixed-prefix-client-summary.json','client-lifecycle.json','graph-before.json','graph-after.json','graph-check.json','prefill-budget.json')]
            names += [f'{arm}/{phase}-rank{rank}.json' for rank in range(4)]
    files={}
    for name in names:
        p=run/name
        if p.is_symlink():raise ValueError('public evidence symlink forbidden')
        if p.is_file():
            raw=p.read_bytes();public_scalars(json.loads(raw));files[name]=raw
    destination.mkdir(parents=True)
    for name,raw in files.items():
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o664)
    result=dict(schema_version=1,status=json.loads(files['experiment.json'])['status'],raw_content_exported=False,
        files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    atomic_json(destination/'manifest.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work-root','records'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--native-baseline-dir',type=Path,default=ROOT/'reports/server/gpqa-c32-9f1635df2afe')
    parser.add_argument('--export-root',type=Path)
    for name,default in [('setup-timeout',7200.),('phase-timeout',360.),('warmup-timeout',960.),('drain-timeout',60.),('gate-timeout',7200.),('cleanup-timeout',120.)]:parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--graph-evidence-port-base',type=int,default=30310)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    for key in ('work_root','records','model_path','native_baseline_dir','export_root'):
        path=getattr(args,key)
        if path is not None:
            if not path.is_absolute():parser.error(key+' must be absolute')
            setattr(args,key,path.resolve())
    if any(not math.isfinite(getattr(args,key)) or getattr(args,key)<=0 for key in ('setup_timeout','phase_timeout','warmup_timeout','drain_timeout','gate_timeout','cleanup_timeout')):parser.error('positive finite timeouts required')
    ports=[args.port]+list(range(args.graph_evidence_port_base,args.graph_evidence_port_base+4))
    if min(ports)<1 or max(ports)>65535 or len(set(ports))!=5:parser.error('distinct valid localhost ports required')
    name='fixed-prefix-'+uuid.uuid4().hex[:12];out=args.work_root/name
    destination=(args.export_root or args.work_root/'public-reports')/name;teacher_path=out/'private/teacher.json';contract=load_contract()
    def launch(arm,checkpoint):return service_command(arm,args.work_root,args.model_path,name,args.port,args.graph_evidence_port_base,checkpoint,out/arm/'private',teacher_path)
    if not args.execute:
        print(json.dumps(dict(status='preview_not_executed',arms=list(ARMS),steps=8,decode_steps=7,batch_size=32,numerical_policy='record_only',
            formal_performance=False,teacher_forcing=True,disable_overlap_schedule=True,services={arm:launch(arm,'0'*64) for arm in ARMS},
            native_trace_steps=2,raw_logits_private=True,prefill_limits=dict(page_size=FIXED_PAGE_SIZE,chunked_prefill_size=FIXED_PREFILL_TOKENS,max_prefill_tokens=FIXED_PREFILL_TOKENS)),indent=2));return 0
    from src.weave.gpqa_c32 import load_gpqa_workload
    from src.weave.checkpoint_identity import checkpoint_stats
    allocation=allocated_gpus();out.mkdir(parents=True,exist_ok=False);teacher_path.parent.mkdir(mode=0o770)
    owner,client_owner=OwnedChild(),OwnedChild();code=1
    report=dict(schema_version=1,status='running',scope='fixed_prefix_native_P0_diagnosis',numerical_policy='record_only',formal_performance=False,
        started_utc=datetime.now(timezone.utc).isoformat(),gpuq_job_id=os.environ.get('GPUQ_JOB_ID'),host_gpu_allocation=allocation.csv,
        steps=8,batch_size=32,disable_overlap_schedule=True,arms={},stage='identity')
    def save():atomic_json(out/'experiment.json',report)
    def cancelled():owner.check_cancelled();client_owner.check_cancelled()
    def on_signal(number,frame):
        for child in (owner,client_owner):child.requested=number
        if not any(c.protected for c in (owner,client_owner)):raise InterruptedError('fixed_prefix_job_cancelled')
    previous={sig:signal.signal(sig,on_signal) for sig in (signal.SIGINT,signal.SIGTERM)}
    def managed(command,folder,lifecycle_name,child_owner,timeout,server=None):
        lifecycle=folder/lifecycle_name
        cmd=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
            '--evidence',str(lifecycle),'--log',str(folder/(lifecycle_name+'.private.log')),'--']+command
        process=child_owner.spawn(cmd,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+timeout
        while process.poll()is None:
            cancelled()
            if server is not None and server.poll()is not None:raise RuntimeError('native service exited during fixed-prefix phase')
            if time.monotonic()>=deadline:raise TimeoutError('fixed-prefix child deadline')
            time.sleep(.2)
        if process.returncode!=0:raise RuntimeError('fixed-prefix child failed; inspect sanitized phase evidence')
        child_owner.stop();cancelled()
    def copy_proofs(arm,phase):
        for rank in range(4):
            path=out/arm/'private'/f'{phase}-rank{rank}.json'
            if path.is_file():atomic_json(out/arm/path.name,_read(path))
    try:
        save();workload=load_gpqa_workload(args.records,model_path=args.model_path);validate_pinned_selection(workload)
        config=prepare_config(workload,steps=8,vocab_size=_read(args.model_path/'config.json')['vocab_size']);atomic_json(out/'config.json',config)
        planned_budget=admission_budget([r['input_tokens'] for r in config['requests']],page_size=FIXED_PAGE_SIZE,chunked_prefill_size=FIXED_PREFILL_TOKENS,max_prefill_tokens=FIXED_PREFILL_TOKENS)
        if not planned_budget['whole_cohort_fits_token_budget']:raise ValueError('planned fixed-prefix prefill budget cannot admit whole cohort')
        atomic_json(out/'prefill-budget.json',dict(planned_budget,configuration_source='planned_fixed_prefix_limits'))
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        source=serving_source_manifest();source_digest=digest(source);atomic_json(out/'source-manifest.json',source)
        checkpoint=native_checkpoint_proof(args.native_baseline_dir,args.model_path);atomic_json(out/'checkpoint-proof.json',checkpoint)
        identity=dict(image_id=image,source_sha256=kernel_source_sha256(),serving_source_sha256=source_digest,
            checkpoint_content_sha256=checkpoint['checkpoint_content_sha256'],native_p0_source_sha256=native_p0_source_sha256(),
            fixed_prefix_source_sha256=fixed_prefix_source_sha256(),observer_source_sha256=graph_observer_source_sha256(),
            graph_sources=_read(ROOT/'configs/sglang_graph_sources.json')['sha256'])
        report['identity']=identity;save()
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32',
             'IMAGE':image,'RUNTIME_PYTHON':contract['python_executable']}
        gate=out/'native-p0-gate';gate.mkdir();cmd=native_p0_gate_command(image,args.work_root,args.model_path,name,allocation.csv)
        lifecycle=gate/'container-lifecycle.json';manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',cmd[cmd.index('--name')+1],'--evidence',str(lifecycle),'--log',str(gate/'private.log'),'--']+cmd
        report['stage']='event_smoke';save();process=owner.spawn(manager,lifecycle,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        wait_graph_gate(process,gate,gate_timeout=args.gate_timeout,cleanup_timeout=args.cleanup_timeout,cancel_check=cancelled);owner.stop();cancelled()
        validate_runtime(_read(gate/'runtime.json'),identity,allocation.csv);smoke=validate_native_p0_smoke(gate,identity);atomic_json(out/'smoke-proof.json',smoke)
        normalized=None;teacher=None
        for arm in ARMS:
            cancelled();folder=out/arm;private=folder/'private';private.mkdir(parents=True,mode=0o770);atomic_json(private/'config.json',config)
            server_dir=args.work_root/('native-ep-'+name+'-'+arm);url=f'http://127.0.0.1:{args.port}'
            group=dict(status='running',arm=arm,phases={});report['arms'][arm]=group;report['stage']='serving_'+arm;save()
            try:
                assert_ports_free(ports)
                with (folder/'launch-private.log').open('w') as log:
                    server=owner.spawn(launch(arm,identity['checkpoint_content_sha256']),server_dir/'container_lifecycle.json',env=env,stdout=log,stderr=subprocess.STDOUT)
                    wait_ready(url,timeout=args.setup_timeout,process=server);validate_runtime(_read(server_dir/'runtime.json'),identity,allocation.csv)
                    actual_launch=_read(server_dir/'launch.json');argv_native=actual_launch['docker_argv']
                    if (actual_launch.get('backend')!='native' or actual_launch.get('graph_mode')!='full' or actual_launch.get('fixed_prefix')is not True
                            or '--disable-overlap-schedule' not in argv_native or '--enable-custom-logit-processor' not in argv_native):raise ValueError('fixed-prefix native launch contract differs')
                    current=normalize_fixed_launch(argv_native)
                    if normalized is None:normalized=current
                    elif current!=normalized:raise ValueError('fixed-prefix launch differs beyond declared arm intervention')
                    atomic_json(out/'launch-comparison.json',dict(status='matched_so_far',checked_arms=list(report['arms']),normalized_sha256=digest(current)))
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    check_prefill_budget(url,config,folder/'prefill-budget.json')
                    atomic_json(folder/'warmup.json',warmup(url,workload,args.warmup_timeout))
                    phases=('seed','measure','trace') if arm=='native_a' else ('measure',)
                    for phase in phases:
                        report['stage']=arm+'_'+phase;save();stage=folder/phase;stage.mkdir();steps=2 if phase=='trace' else 8
                        flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                        check_prefill_budget(url,config,stage/'prefill-budget.json')
                        before=baseline_coverage(args.graph_evidence_port_base,deadline=time.monotonic()+30.,endpoint='graph-coverage');atomic_json(stage/'graph-before.json',before)
                        command=phase_command(image,args.work_root,args.records,f'{name}/{arm}/{phase}',f'{name}/private/teacher.json',name+'-'+arm+'-'+phase,phase,args.port,args.phase_timeout)
                        try:
                            managed(command,stage,'client-lifecycle.json',client_owner,args.phase_timeout+60.,server)
                        finally:copy_proofs(arm,phase)
                        client=_read(stage/'fixed-prefix-client-summary.json')
                        if (client.get('status')!='passed' or client.get('requests_completed')!=32 or client.get('cleanup_complete')is not True
                                or client.get('phase')!=phase or client.get('steps')!=steps or client.get('selection',{}).get('selection_sha256')!=config['selection_sha256']):raise ValueError('complete matching fixed-prefix client responses required')
                        teacher=load_teacher(teacher_path,selection_sha256=config['selection_sha256'],steps=8,vocab_size=config['vocab_size'])
                        atomic_json(out/'teacher-manifest.json',{k:v for k,v in teacher.items() if k!='output_ids'})
                        wait_phase_proofs(private,phase=phase,steps=steps,config=config,teacher=teacher,identity=identity,timeout=args.drain_timeout,server=server,cancel_check=cancelled)
                        copy_proofs(arm,phase)
                        flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                        after=baseline_coverage(args.graph_evidence_port_base,deadline=time.monotonic()+30.,endpoint='graph-coverage');atomic_json(stage/'graph-after.json',after)
                        graph=validate_fixed_graph(before,after,arm=arm,steps=steps,identity=identity);atomic_json(stage/'graph-check.json',graph)
                        group['phases'][phase]={'status':'passed','graph':graph};save()
                owner.stop();cancelled();group['status']='passed'
            except BaseException as error:
                group.update(status='failed',error_type=type(error).__name__);raise
            finally:
                for phase in ('seed','measure','trace'):copy_proofs(arm,phase)
                for src,dst in [('launch.json','launch.json'),('runtime.json','runtime.json'),('container_lifecycle.json','server-lifecycle.json')]:
                    if (server_dir/src).is_file():atomic_json(folder/dst,_read(server_dir/src))
                atomic_json(folder/'group.json',group);save()
        launch_check=_read(out/'launch-comparison.json');launch_check['status']='matched';atomic_json(out/'launch-comparison.json',launch_check)
        report['stage']='cpu_comparison';save()
        command=cpu_container(image,args.work_root,name+'-compare')+[image,contract['python_executable'],'/workspace/scripts/compare_fixed_prefix.py',
            '--root',f'/work/{name}','--teacher',f'/work/{name}/private/teacher.json','--output',f'/work/{name}/comparison.json','--execute']
        managed(command,out,'comparison-lifecycle.json',client_owner,1800.)
        comparison=_read(out/'comparison.json')
        if comparison.get('status')!='comparison_complete' or comparison.get('teacher_sha256')!=teacher['sha256']:raise ValueError('fixed-prefix CPU comparison incomplete')
        if digest(serving_source_manifest())!=source_digest:raise ValueError('source changed during fixed-prefix run')
        if load_gpqa_workload(args.records,model_path=args.model_path).identity!=workload.identity:raise ValueError('prepared data changed')
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed')
        if validate_native_p0_smoke(gate,identity)!=smoke:raise ValueError('event smoke changed')
        report.update(status='diagnostic_complete',stage='complete');code=0
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__)
        p=out/'controller-error-private.log';p.write_text(str(error)+'\n');p.chmod(0o600)
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={name:pool.submit(child.stop) for name,child in [('server',owner),('client',client_owner)]}
            for label,future in futures.items():
                try:future.result()
                except BaseException as error:report.update(status='failed',cleanup_error_type=type(error).__name__);code=1
        for arm in report['arms']:
            path=args.work_root/('native-ep-'+name+'-'+arm)/'container_lifecycle.json'
            if path.is_file():atomic_json(out/arm/'server-lifecycle.json',_read(path))
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        try:export_public(out,destination)
        except BaseException as error:
            report.update(status='failed',export_error_type=type(error).__name__);code=1;save()
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(dict(status=report['status'],work_directory=str(out),export_directory=str(destination))),flush=True);return code


if __name__=='__main__':raise SystemExit(main())
