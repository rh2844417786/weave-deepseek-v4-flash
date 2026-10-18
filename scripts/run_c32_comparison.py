#!/usr/bin/env python3
"""Owned offline C32 eager comparison: native, cached persistent, staged direct."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_weave import validate_core_evidence
from scripts.check_sglang_service import wait_ready,coverage,http_json
from scripts.run_concurrency_diagnostic import validate_dataset
from scripts.run_model_acceptance import OwnedChild,assert_ports_free
from src.weave.calibration import atomic_json
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.c32_comparison import (VARIANTS,HOPPER_VARIANTS,PREFILL_VARIANTS,digest,compare_acceptance,validate_coverage_delta,
    validate_client_summary,scheduler_counts,summarize_comparison,validate_online_gate)
from src.weave.c32_serving import DATASET_SHA256
from src.weave.experiment_export import ensure_public
from src.weave.gpu_allocation import allocated_gpus,GPUAllocation
from src.weave.hopper_candidate import source_sha256 as candidate_source_sha256
from src.weave.hopper_decode import assembly_source_sha256
from src.weave.hopper_serving_evidence import (REPLAY_FILES,validate_replay_bundle,replay_bundle_bytes,
    record_generation_comparison,validate_generation_proof,validate_assembly_reports)
from src.weave.ops import kernel_source_sha256,kernel_source_manifest
from src.weave.performance import flush_cache_when_idle
from src.weave.replay_trace import tokenizer_manifest_sha256
from src.weave.runtime_contract import load_contract


def serving_source_manifest():
    paths=set()
    for directory,pattern in (('scripts','*.py'),('src/weave','*.py'),('tests/gpu','*.py'),
                              ('configs','*.json'),('timing_hooks','*.py')):
        paths.update(p for p in (ROOT/directory).rglob(pattern) if p.is_file())
    return {'schema_version':1,'files':[{'path':p.relative_to(ROOT).as_posix(),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]}


def service_command(variant,work,model,run_name,port,base,checkpoint,*,stall_diagnostic=False,stall_port_base=30210):
    if variant not in set(VARIANTS)|set(HOPPER_VARIANTS)|set(PREFILL_VARIANTS):raise ValueError('unknown C32 group')
    result = [sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--execute',
        '--backend','native' if variant=='native' else 'weave_functional',
        '--kernel','tiled','--decode-graph','disabled','--fixed-plan','1,1,0','--hot-cache',
        '--decode-implementation',('hopper_fp8' if variant=='hopper_prefill' else variant) if variant in ('staged_direct','hopper_fp8','hopper_prefill') else 'persistent',
        '--prefill-implementation','hopper_fp8' if variant=='hopper_prefill' else 'persistent',
        '--request-timing','--work-root',str(work),'--model-path',str(model),
        '--run-name',run_name,'--port',str(port),'--evidence-port-base',str(base),
        '--checkpoint-content-sha256',checkpoint]
    if stall_diagnostic:result += ['--stall-diagnostic','--stall-port-base',str(stall_port_base)]
    return result


def client_command(image,work,dataset,output_relative,name,port,seconds,acceptance,*,stall_diagnostic=False,stall_port_base=30210):
    result=['docker','run','--rm','--pull=never','--name','weave-'+name,'--network=host',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work',
        '-v',f'{dataset}:/dataset.jsonl:ro',image,load_contract()['python_executable'],
        '/workspace/scripts/bench_c32_serving.py','--dataset','/dataset.jsonl',
        '--base-url',f'http://127.0.0.1:{port}','--output-dir',f'/work/{output_relative}',
        '--seconds',str(seconds),'--concurrency','32','--output-tokens','1024','--execute']
    if acceptance:result+=['--acceptance-only']
    elif stall_diagnostic:result+=['--stall-diagnostic','--stall-port-base',str(stall_port_base)]
    return result


def gate_command(image,work,model,run_name,allocation,comparison_suite='staged',*,stall_diagnostic=False):
    if comparison_suite not in ('staged','hopper','prefill'):raise ValueError('unknown comparison suite')
    devices=GPUAllocation.parse(allocation);python=load_contract()['python_executable']
    command=['docker','run','--rm','--pull=never','--name','weave-online-gate-'+run_name,
        '--gpus',devices.docker_argument,'--network=host','--shm-size=8g',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work','-v',f'{model}:/model:ro',
        '-e',f'CUDA_VISIBLE_DEVICES={devices.container_visible}','-e','GLOO_SOCKET_IFNAME=lo',
        '-e','NCCL_IB_DISABLE=1','-e','NCCL_SOCKET_IFNAME=lo','-e','NCCL_CUMEM_HOST_ENABLE=0',
        '-e','TORCH_NCCL_ASYNC_ERROR_HANDLING=1','-e','TORCH_NCCL_BLOCKING_WAIT=1',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','TORCH_EXTENSIONS_DIR=/work/torch_extensions','-e','MAX_JOBS=2',
        '-e',f'WEAVE_HOST_ALLOCATION={devices.csv}','-e',f'WEAVE_IMAGE_ID={image}',
        '-e',f'WEAVE_IMAGE_REF={load_contract()["image"]}','-e','PYTHONUNBUFFERED=1',
        image,python,'/workspace/scripts/check_runtime.py','--stage','native',
        '--output-dir',f'/work/{run_name}/online-gate','--exec-command',python,
        '-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        '/workspace/tests/gpu/check_online_staged.py','--model-path','/model',
        '--output-dir',f'/work/{run_name}/online-gate']
    if comparison_suite=='prefill':
        offset=command.index('--exec-command')+1
        command[offset:]=[python,'/workspace/scripts/run_prefill_gate.py','--model-path','/model','--output-dir',f'/work/{run_name}/online-gate']
        if stall_diagnostic:command+=['--stall-diagnostic']
        return command
    if comparison_suite=='hopper':
        offset=command.index('/workspace/tests/gpu/check_online_staged.py')
        command[offset:]=['/workspace/tests/gpu/check_hopper_serving.py','--output-dir',f'/work/{run_name}/online-gate']
    if stall_diagnostic:
        if comparison_suite!='hopper':raise ValueError('stall diagnostic requires Hopper suite')
        offset=command.index('--exec-command')+1
        command[offset:]=[python,'/workspace/scripts/run_stall_gate.py','--output-dir',f'/work/{run_name}/online-gate']
    return command


def validate_stall_smoke(folder):
    for rank in range(4):
        value=json.loads((folder/f'stall-smoke-rank{rank}.json').read_text())
        if value.get('rank')!=rank or value.get('status')!='passed':raise ValueError('stall_smoke_incomplete')


def snapshot_stalls(folder,base,*,expected_source=None,expected_image=None,expected_candidate=None):
    """Called only after abort/drain/idle; snapshot GPU reads never enter measurement."""
    from src.weave.stall_analysis import analyze_stalls
    reports=[]
    for rank in range(4):
        value=http_json(f'http://127.0.0.1:{base+rank}/snapshot',timeout=120)
        if value.get('rank')!=rank:raise ValueError('stall_rank_identity_differs')
        if ((expected_source is not None and value.get('source_sha256')!=expected_source)
                or (expected_image is not None and value.get('identity',{}).get('image')!=expected_image)
                or (expected_candidate is not None and value.get('identity',{}).get('candidate')!=expected_candidate)):
            raise ValueError('stall_source_identity_differs')
        value.update(instrumented_diagnostic=True,used_for_formal_performance=False)
        atomic_json(folder/f'stall-rank{rank}.json',value);reports.append(value)
    analysis=analyze_stalls(reports,json.loads((folder/'client-stall.json').read_text()),
        json.loads((folder/'load-window.json').read_text()))
    analysis.update(instrumented_diagnostic=True,used_for_formal_performance=False)
    atomic_json(folder/'stall-analysis.json',analysis)
    if analysis.get('status')!='passed':raise ValueError('stall_diagnostic_incomplete')


def export_public(run,destination):
    run,destination=Path(run),Path(destination)
    if destination.exists():raise FileExistsError('fresh public destination required')
    names=['experiment.json','comparison.json','source-manifest.json','serving-source-manifest.json',
           'online-gate/runtime.json','online-gate/container_lifecycle.json',
           'online-gate/checkpoint_identity.json','online-gate/gate-check.json']
    names += ['replay-proof-check.json']
    names += [f'online-gate/hopper-serving-rank{r}.json' for r in range(4)]
    names += [f'online-gate/hopper-prefill-rank{r}.json' for r in range(4)]
    names += [f'online-gate/stall-smoke-rank{r}.json' for r in range(4)]
    names += [f'replay-proof/{name}' for name in sorted(REPLAY_FILES|{'manifest.json'})]
    names += [f'online-gate/online-staged-rank{r}.json' for r in range(4)]
    names += [f'online-gate/online-staged-progress-rank{r}.json' for r in range(4)]
    group_names=('group.json','launch.json','runtime.json','server-lifecycle.json','acceptance-lifecycle.json',
        'measurement-lifecycle.json','acceptance-summary.json','acceptance-check.json','client-summary.json',
        'load-window.json','load-progress.json','coverage-before.json','coverage-after.json',
        'coverage-check.json','scheduler-summary.json','client-stall.json','stall-analysis.json',
        *(f'stall-rank{r}.json' for r in range(4)))
    names += [f'{group}/{name}' for group in sorted(set(VARIANTS)|set(HOPPER_VARIANTS)|set(PREFILL_VARIANTS)) for name in group_names]
    files={name:(run/name).read_bytes() for name in names if (run/name).is_file()}
    report=json.loads(files['experiment.json'])
    if report.get('core_run_dir'):
        core=Path(report['core_run_dir'])
        for name in ['runtime.json','checkpoint_identity.json']+[f'{prefix}{r}.json'
                    for prefix in ('rank','checkpoint-rank') for r in range(4)]:
            if (core/name).is_file():files['core/'+name]=(core/name).read_bytes()
    for name,raw in files.items():
        if name=='replay-proof/compilation-resources.txt':
            if any(line and not line.lstrip().startswith(('ptxas info','ptxas warning')) for line in raw.decode().splitlines()):
                raise ValueError('only compiler resource lines may be exported')
            continue
        value=json.loads(raw);ensure_public(value)
        if any(key in json.dumps(value) for key in ('"output_token_ids":','"records":')):
            raise ValueError('private acceptance records cannot be exported')
    destination.mkdir(parents=True)
    manifest={'schema_version':1,'status':report['status'],'raw_content_exported':False,'files':{}}
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o664)
        manifest['files'][name]=hashlib.sha256(raw).hexdigest()
    atomic_json(destination/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--core-run-dir',type=Path)
    parser.add_argument('--comparison-suite',choices=('staged','hopper','prefill'),default='staged')
    parser.add_argument('--hopper-evidence-dir',type=Path,help='immutable complete schema2 public 1376-point Hopper replay bundle')
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--gate-timeout',type=float,default=7200.)
    parser.add_argument('--acceptance-timeout',type=float,default=1200.)
    parser.add_argument('--drain-timeout',type=float,default=600.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--stall-diagnostic',action='store_true')
    parser.add_argument('--stall-port-base',type=int,default=30210)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.stall_diagnostic:
        if args.comparison_suite not in ('hopper','prefill'):parser.error('stall diagnostic only supports Hopper suite')
        ports=[args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4))+list(range(args.stall_port_base,args.stall_port_base+4))
        if not 1<=args.stall_port_base<=65532 or len(set(ports))!=9:parser.error('stall ports must be valid and distinct')
    diagnostic_options={'stall_diagnostic':args.stall_diagnostic,'stall_port_base':args.stall_port_base}
    for key in ('work_root','dataset','core_run_dir','model_path','export_root','hopper_evidence_dir'):
        value=getattr(args,key)
        if value is not None:
            if not value.is_absolute():parser.error(f'{key} must be absolute')
            setattr(args,key,value.resolve())
    if args.comparison_suite in ('hopper','prefill') and args.hopper_evidence_dir is None:parser.error('Hopper suite requires --hopper-evidence-dir')
    if args.core_run_dir is None:
        if args.comparison_suite=='staged':parser.error('staged suite requires --core-run-dir')
        args.core_run_dir=args.hopper_evidence_dir/'core'
    variants=PREFILL_VARIANTS if args.comparison_suite=='prefill' else HOPPER_VARIANTS if args.comparison_suite=='hopper' else VARIANTS
    if args.comparison_suite=='prefill' and args.seconds!=600.:parser.error('prefill comparison requires the unchanged 600-second group window')
    if any(not math.isfinite(v) or v<=0 for v in (args.seconds,args.setup_timeout,args.gate_timeout,args.acceptance_timeout,args.drain_timeout)):
        parser.error('finite positive budgets required')
    if (not 1<=args.port<=65535 or not 1<=args.evidence_port_base<=65532
            or args.port in range(args.evidence_port_base,args.evidence_port_base+4)):
        parser.error('valid distinct serving/evidence ports required')
    allocation=allocated_gpus();contract=load_contract();name='c32-'+args.comparison_suite+'-'+uuid.uuid4().hex[:12]
    out=args.work_root/name;destination=(args.export_root or args.work_root/'public-reports')/name
    preview={v:service_command(v,args.work_root,args.model_path,'native-ep-'+name+'-'+v,
                 args.port,args.evidence_port_base,'0'*64,**diagnostic_options) for v in variants}
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','comparison_suite':args.comparison_suite,'groups':preview,
            'hopper_evidence_dir':str(args.hopper_evidence_dir) if args.hopper_evidence_dir else None,
            'generation_comparison_policy':'record_only' if args.comparison_suite in ('hopper','prefill') else 'legacy_exact',
            'required_environment':{'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'},
            'online_gate':gate_command(contract['image'],args.work_root,args.model_path,name,allocation.csv,args.comparison_suite,stall_diagnostic=args.stall_diagnostic)},indent=2));return 0
    out.mkdir(parents=True,exist_ok=False)
    owner=OwnedChild();client_owner=OwnedChild();code=1
    report={'schema_version':1,'scope':args.comparison_suite+'_C32_eager_serving','comparison_suite':args.comparison_suite,'status':'running','stage':'validating',
        'started_utc':datetime.now(timezone.utc).isoformat(),'core_run_dir':str(args.core_run_dir),
        'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),'host_gpu_allocation':allocation.csv,
        'kernel_source_sha256':kernel_source_sha256(),'serving_source_sha256':digest(serving_source_manifest()),
        'formal_gpqa':False,'groups':{}}
    if args.stall_diagnostic:report.update(instrumented_diagnostic=True,used_for_formal_performance=False)
    if args.comparison_suite in ('hopper','prefill'):report.update(numerical_policy='record_only',generation_comparison_policy='record_only')
    def save():atomic_json(out/'experiment.json',report)
    def cancel(number,frame):
        for child in (owner,client_owner):child.requested=number
        if not any(child.protected for child in (owner,client_owner)):raise InterruptedError('job_cancelled')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    def check_cancelled():owner.check_cancelled();client_owner.check_cancelled()
    def managed(command,folder,label,timeout,*,service=None,measurement=False):
        lifecycle=folder/f'{label}-lifecycle.json'
        manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
            '--evidence',str(lifecycle),'--log',str(folder/f'{label}-private.log'),'--']+command
        child=client_owner.spawn(manager,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        started=time.monotonic();deadline=started+timeout
        while child.poll() is None:
            check_cancelled()
            if service is not None and service.poll() is not None:raise RuntimeError('service_exited_during_client')
            if measurement and (folder/'load-window.json').is_file():
                marker=json.loads((folder/'load-window.json').read_text())
                end=marker.get('deadline_monotonic');begin=marker.get('started_monotonic')
                if (type(end) not in (float,int) or type(begin) not in (float,int)
                        or not started<=begin<=time.monotonic() or not math.isclose(end-begin,args.seconds,rel_tol=1e-9)):
                    raise ValueError('invalid_client_window_marker')
                deadline=end+args.drain_timeout+30
            if time.monotonic()>=deadline:raise TimeoutError(label+'_deadline')
            time.sleep(.2)
        returncode=child.returncode;client_owner.stop();check_cancelled()
        if returncode!=0:raise RuntimeError(label+'_client_failed')
        return lifecycle
    try:
        save();dataset=validate_dataset(args.dataset)
        if dataset['sha256']!=DATASET_SHA256:raise ValueError('existing_C32_dataset_SHA256_required')
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        gate,checkpoint=validate_core_evidence(args.core_run_dir,args.model_path,report['kernel_source_sha256'],{'resolved_image_id':image})
        identity={'dataset_sha256':dataset['sha256'],'image_id':image,'source_sha256':report['kernel_source_sha256'],
            'serving_source_sha256':report['serving_source_sha256'],'checkpoint_content_sha256':gate['checkpoint_content_sha256'],
            'tokenizer_manifest_sha256':tokenizer_manifest_sha256(args.model_path),'seconds':args.seconds,
            'decode_graph':'disabled','concurrency':32,'max_new_tokens':1024,'temperature':0,'ignore_eos':True,
            'host_gpu_allocation':allocation.csv}
        if args.stall_diagnostic:
            from src.weave.stall_hooks import stall_source_sha256
            identity.update(instrumented_diagnostic=True,used_for_formal_performance=False,stall_source_sha256=stall_source_sha256())
        if args.comparison_suite in ('hopper','prefill'):
            replay_current={'candidate_source_sha256':candidate_source_sha256(),'bf16_source_sha256':identity['source_sha256'],
                'checkpoint_content_sha256':identity['checkpoint_content_sha256'],'image_id':image,
                'tokenizer_manifest_sha256':identity['tokenizer_manifest_sha256']}
            replay_check=validate_replay_bundle(args.hopper_evidence_dir,**replay_current)
            identity.update(candidate_source_sha256=replay_current['candidate_source_sha256'],
                assembly_source_sha256=assembly_source_sha256(),replay_manifest_sha256=replay_check['manifest_sha256'],
                numerical_policy='record_only',generation_comparison_policy='record_only')
            if args.comparison_suite=='prefill':
                from src.weave.hopper_prefill_candidate import source_sha256 as prefill_source
                from src.weave.hopper_prefill import runtime_source_sha256
                identity.update(prefill_source_sha256=prefill_source(),prefill_runtime_source_sha256=runtime_source_sha256())
            atomic_json(out/'replay-proof-check.json',replay_check)
            for relative,raw in replay_bundle_bytes(args.hopper_evidence_dir).items():
                path=out/'replay-proof'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            if validate_replay_bundle(out/'replay-proof',**replay_current)!=replay_check:
                raise ValueError('replay_changed_while_copying')
        report.update(identity=identity,core_gate=gate,stage='hopper_assembly_gate' if args.comparison_suite in ('hopper','prefill') else 'online_staged_gate');save()
        atomic_json(out/'source-manifest.json',kernel_source_manifest());atomic_json(out/'serving-source-manifest.json',serving_source_manifest())
        gate_folder=out/'online-gate';gate_folder.mkdir()
        managed(gate_command(image,args.work_root,args.model_path,name,allocation.csv,args.comparison_suite,stall_diagnostic=args.stall_diagnostic),gate_folder,'container',args.gate_timeout)
        if args.stall_diagnostic:validate_stall_smoke(gate_folder)
        # Preserve the manager's public name consistently with other gates.
        atomic_json(gate_folder/'container_lifecycle.json',json.loads((gate_folder/'container-lifecycle.json').read_text()))
        if args.comparison_suite in ('hopper','prefill'):
            gate_reports=[json.loads((gate_folder/f'hopper-serving-rank{r}.json').read_text()) for r in range(4)]
            online_check=validate_assembly_reports(gate_reports,candidate_source_sha256=identity['candidate_source_sha256'],
                bf16_source_sha256=identity['source_sha256'],assembly_source_sha256=identity['assembly_source_sha256'],image_id=image)
        else:
            gate_reports=[json.loads((gate_folder/f'online-staged-rank{r}.json').read_text()) for r in range(4)]
            online_check=validate_online_gate(gate_reports,identity['source_sha256'],gate['checkpoint_content_sha256'])
        if args.comparison_suite=='prefill':
            from src.weave.hopper_prefill_evidence import validate_prefill_reports
            online_check['prefill']=validate_prefill_reports([json.loads((gate_folder/f'hopper-prefill-rank{r}.json').read_text()) for r in range(4)],
                **{key:identity[key] for key in ('prefill_source_sha256','prefill_runtime_source_sha256','candidate_source_sha256','checkpoint_content_sha256','image_id')},bf16_source_sha256=identity['source_sha256'])
        atomic_json(gate_folder/'gate-check.json',online_check)
        native_proof=None
        env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
            'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':contract['image'],'RUNTIME_PYTHON':contract['python_executable']}
        for variant in variants:
            check_cancelled();report.update(stage='serving_'+variant);save()
            folder=out/variant;folder.mkdir();run_name='native-ep-'+name+'-'+variant
            server_dir=args.work_root/run_name;url=f'http://127.0.0.1:{args.port}'
            group={'status':'running','identity':identity,'variant':variant,'decode_implementation':
                ('hopper_fp8' if variant=='hopper_prefill' else variant) if variant in ('staged_direct','hopper_fp8','hopper_prefill') else 'persistent','prefill_implementation':'hopper_fp8' if variant=='hopper_prefill' else 'persistent','stage':'starting'}
            if args.stall_diagnostic:group.update(instrumented_diagnostic=True,used_for_formal_performance=False)
            try:
                assert_ports_free([args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4))
                    +(list(range(args.stall_port_base,args.stall_port_base+4)) if args.stall_diagnostic else []))
                with (folder/'server-launch-private.log').open('w') as log:
                    server=owner.spawn(service_command(variant,args.work_root,args.model_path,run_name,args.port,
                        args.evidence_port_base,gate['checkpoint_content_sha256'],**diagnostic_options),server_dir/'container_lifecycle.json',
                        env=env,stdout=log,stderr=subprocess.STDOUT)
                    wait_ready(url,timeout=args.setup_timeout,process=server)
                    runtime=json.loads((server_dir/'runtime.json').read_text())
                    if (runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=image
                            or runtime.get('kernel_source_sha256')!=identity['source_sha256']
                            or runtime.get('host_gpu_allocation')!=allocation.csv):raise ValueError('service_runtime_identity_differs')
                    group['stage']='acceptance';atomic_json(folder/'group.json',group)
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    managed(client_command(image,args.work_root,args.dataset,f'{name}/{variant}',name+'-'+variant+'-accept',
                        args.port,args.seconds,True),folder,'acceptance',args.acceptance_timeout,service=server)
                    proof=json.loads((folder/'acceptance-private.json').read_text());accept=json.loads((folder/'acceptance-summary.json').read_text())
                    if (accept.get('status')!='passed' or accept.get('dataset_sha256')!=dataset['sha256']
                            or accept.get('requests_completed')!=32 or accept.get('generated_tokens')!=256
                            or accept.get('private_artifact_sha256')!=hashlib.sha256((folder/'acceptance-private.json').read_bytes()).hexdigest()):
                        raise ValueError('acceptance_proof_incomplete')
                    if native_proof is None:native_proof=proof
                    if args.comparison_suite in ('hopper','prefill'):
                        validate_generation_proof(accept,proof,
                            private_sha256=hashlib.sha256((folder/'acceptance-private.json').read_bytes()).hexdigest(),
                            dataset_sha256=dataset['sha256'])
                        if any(native_proof.get(k)!=proof.get(k) for k in ('dataset_sha256','request_order_sha256')):
                            raise ValueError('generation_workload_identity_changed')
                    compare=record_generation_comparison if args.comparison_suite in ('hopper','prefill') else compare_acceptance
                    group['acceptance']=compare(native_proof,proof);atomic_json(folder/'acceptance-check.json',group['acceptance'])
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    before=coverage(args.evidence_port_base) if variant!='native' else None
                    if before is not None:atomic_json(folder/'coverage-before.json',before)
                    log_offset=(server_dir/'server.log').stat().st_size
                    group['stage']='measurement';atomic_json(folder/'group.json',group)
                    managed(client_command(image,args.work_root,args.dataset,f'{name}/{variant}',name+'-'+variant+'-load',
                        args.port,args.seconds,False,**diagnostic_options),folder,'measurement',args.setup_timeout,service=server,measurement=True)
                    client=json.loads((folder/'client-summary.json').read_text());validate_client_summary(client,args.seconds,dataset['sha256'])
                    if args.stall_diagnostic:
                        client.update(instrumented_diagnostic=True,used_for_formal_performance=False)
                        atomic_json(folder/'client-summary.json',client)
                    if client['request_order_sha256']!=accept['request_order_sha256']:raise ValueError('client_order_changed')
                    http_json(url+'/abort_request',{'abort_all':True},timeout=30)
                    flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,max_wait=args.drain_timeout)
                    if args.stall_diagnostic:
                        snapshot_stalls(folder,args.stall_port_base,expected_source=identity['stall_source_sha256'],
                            expected_image=image,expected_candidate=variant)
                    with (server_dir/'server.log').open('rb') as stream:stream.seek(log_offset);logs=stream.read().decode(errors='replace')
                    scheduler=scheduler_counts(logs);atomic_json(folder/'scheduler-summary.json',scheduler)
                    if variant!='native':
                        after=coverage(args.evidence_port_base);atomic_json(folder/'coverage-after.json',after)
                        checked=validate_coverage_delta(before,after,group['decode_implementation'],identity['source_sha256'],gate['checkpoint_content_sha256'],
                            **({'candidate_source_sha256':identity['candidate_source_sha256'],'assembly_source_sha256':identity['assembly_source_sha256']} if variant in ('hopper_fp8','hopper_prefill') else {}),
                            **({'prefill_implementation':'hopper_fp8' if variant=='hopper_prefill' else 'persistent',
                                'prefill_source_sha256':identity['prefill_source_sha256'],
                                'prefill_runtime_source_sha256':identity['prefill_runtime_source_sha256']} if args.comparison_suite=='prefill' else {}))
                        atomic_json(folder/'coverage-check.json',checked)
                        batch={'max_running_requests':32,'evidence':'all_rank_adapter_decode_batch_histograms',
                            'rank0_batch32_forwards':checked['decode_batch32_forwards_rank0']}
                    else:
                        if scheduler['max_running_requests']!=32:raise ValueError('native_decode_concurrency32_not_observed')
                        batch=scheduler
                    group.update(status='passed',stage='complete',client=client,batch_evidence=batch)
                owner.stop();check_cancelled()
            except BaseException as exc:
                group.update(status='failed',error_type=type(exc).__name__)
                raise
            finally:
                atomic_json(folder/'group.json',group);report['groups'][variant]=group
                for src,dst in (('launch.json','launch.json'),('runtime.json','runtime.json'),('container_lifecycle.json','server-lifecycle.json')):
                    if (server_dir/src).is_file():atomic_json(folder/dst,json.loads((server_dir/src).read_text()))
                save()
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint_changed')
        if validate_dataset(args.dataset)!=dataset:raise ValueError('dataset_changed')
        if tokenizer_manifest_sha256(args.model_path)!=identity['tokenizer_manifest_sha256']:raise ValueError('tokenizer_changed')
        if kernel_source_sha256()!=identity['source_sha256'] or digest(serving_source_manifest())!=identity['serving_source_sha256']:
            raise ValueError('source_changed_during_run')
        if args.comparison_suite in ('hopper','prefill'):
            if (candidate_source_sha256()!=identity['candidate_source_sha256']
                    or assembly_source_sha256()!=identity['assembly_source_sha256']
                    or validate_replay_bundle(args.hopper_evidence_dir,**replay_current)!=replay_check):
                raise ValueError('hopper_source_or_replay_changed_during_run')
        if args.comparison_suite=='prefill' and (prefill_source()!=identity['prefill_source_sha256'] or runtime_source_sha256()!=identity['prefill_runtime_source_sha256']):raise ValueError('prefill_source_changed_during_run')
        comparison=summarize_comparison(report['groups'],comparison_suite=args.comparison_suite)
        if args.stall_diagnostic:
            comparison={k:v for k,v in comparison.items() if not k.endswith('_ratio')}
            comparison.update(instrumented_diagnostic=True,used_for_formal_performance=False,
                scope='instrumented C32 stall diagnostic; rates are diagnostic observations')
        atomic_json(out/'comparison.json',comparison)
        report.update(status='comparison_complete',stage='complete');code=0
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        (out/'controller-error-private.log').write_text(str(exc)+'\n')
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={label:pool.submit(child.stop) for label,child in (('server',owner),('client',client_owner))}
            for label,future in pending.items():
                try:future.result()
                except BaseException as exc:report.update(status='failed',cleanup_error_type=type(exc).__name__);code=1
        # Copy lifecycle after cancellation cleanup too.
        for variant in report['groups']:
            server_dir=args.work_root/('native-ep-'+name+'-'+variant)
            if (server_dir/'container_lifecycle.json').is_file():
                atomic_json(out/variant/'server-lifecycle.json',json.loads((server_dir/'container_lifecycle.json').read_text()))
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),export_directory=str(destination));save()
        export_public(out,destination)
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'work_directory':str(out),'export_directory':str(destination)}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
