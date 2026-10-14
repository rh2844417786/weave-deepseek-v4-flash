#!/usr/bin/env python3
"""Capture two private real C32 decode steps with the installed offline workload.

The owned service is aborted/drained before device snapshots are downloaded.
Only allowlisted manifests are copied to public output. No timing from this
job is a performance result. Requires a passed current-source all43 core gate.
"""
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
from scripts.calibrate_weave import validate_core_evidence
from scripts.check_sglang_service import wait_ready,coverage,http_json
from scripts.run_concurrency_diagnostic import validate_dataset
from scripts.run_model_acceptance import OwnedChild,assert_ports_free
from src.weave.calibration import atomic_json
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.gpu_allocation import allocated_gpus
from src.weave.ops import kernel_source_sha256
from src.weave.performance import flush_cache_when_idle
from src.weave.replay_trace import (aggregate_manifests,capture_source_sha256,
    validate_capture_directory,validate_identity,validate_reused_capture,
    tokenizer_manifest,tokenizer_manifest_sha256)
from src.weave.runtime_contract import load_contract


def installed_benchmark(arguments):
    """Run exactly the audited installed benchmark; never reset profile buffers."""
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
    from sglang.benchmark import serving
    from sglang.benchmark.datasets import custom
    expected={serving:'73a7ce619e8d4c6539d89755f465beedfc41fff9',
              custom:'452c7db74c0f1aefd2792c6308f3b9f9a3f28e93'}
    for module,git_blob in expected.items():
        payload=Path(module.__file__).read_bytes()
        actual=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
        if actual!=git_blob:
            raise ValueError('installed SGLang benchmark differs from audited commit2948168; use the existing audited image')
    sys.argv=['sglang.benchmark.serving']+arguments
    serving.cli_main()
    return 0


def load_command(image,work,model,dataset,run_name,port):
    return ['docker','run','--rm','--pull=never','--name','weave-replay-load-'+run_name,
        '--network=host','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '-e','HF_DATASETS_OFFLINE=1','-e','PYTHONUNBUFFERED=1',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work','-v',f'{model}:/model:ro',
        '-v',f'{dataset}:/dataset.jsonl:ro',image,load_contract()['python_executable'],
        '/workspace/scripts/run_replay_capture.py','--installed-benchmark',
        '--backend','sglang','--base-url',f'http://127.0.0.1:{port}',
        '--model','/model','--tokenizer','/model','--served-model-name','weave-dsv4-functional',
        '--dataset-name','custom','--dataset-path','/dataset.jsonl','--num-prompts','1800',
        '--sharegpt-output-len','1024','--max-concurrency','32','--request-rate','inf',
        '--temperature','0','--warmup-requests','1','--disable-tqdm',
        '--output-file',f'/work/{run_name}/bench-private.jsonl']


def validate_service(values,identity):
    if len(values)!=4:raise ValueError('four service coverage ranks required')
    for rank,value in enumerate(values):
        if (value.get('rank')!=rank or value.get('kernel')!='tiled'
                or value.get('graph_enabled') is not False or value.get('schedule_mode')!='explicit_fixed'
                or value.get('fixed_plan')!=[1,1,0] or value.get('profile_bank_sha256') is not None
                or value.get('execution_profile_configuration') is not None
                or value.get('source_sha256')!=identity['kernel_source_sha256']
                or value.get('profile_checkpoint_content_sha256')!=identity['checkpoint_content_sha256']
                or value.get('hot_cache') is not True
                or set(value.get('layers',{}))!={str(i) for i in range(43)}):
            raise ValueError('capture service differs from all43 tiled fixed/eager contract')


def capture_status(base,identity):
    values=[]
    for rank in range(4):
        value=http_json(f'http://127.0.0.1:{base+rank}/replay-capture/status',timeout=10)
        if value.get('rank')!=rank:raise ValueError('capture status rank mismatch')
        validate_identity(value.get('identity'),identity)
        if value.get('coverage',{}).get('failed'):raise ValueError('model forward failed during capture')
        values.append(value)
    return values


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True,help='existing9-case x200 custom JSONL; never regenerated')
    parser.add_argument('--core-run-dir',type=Path,required=True,help='passed fresh-source all43/three-kernel checkpoint gate')
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path,help='public allowlisted manifests only')
    parser.add_argument('--reuse-capture-dir',type=Path,help='private completed trace; hashes and current identity must match')
    parser.add_argument('--run-name',help='fresh replay-capture-* directory name')
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--setup-timeout',type=float,default=7200.)
    parser.add_argument('--capture-timeout',type=float,default=7200.)
    parser.add_argument('--drain-timeout',type=float,default=600.)
    parser.add_argument('--persist-timeout',type=float,default=600.)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    for field in ('work_root','dataset','core_run_dir','model_path','export_root','reuse_capture_dir'):
        path=getattr(args,field)
        if path is not None:
            if not path.is_absolute():parser.error(f'{field} must be absolute')
            setattr(args,field,path.resolve())
    for field in ('setup_timeout','capture_timeout','drain_timeout','persist_timeout'):
        if not math.isfinite(getattr(args,field)) or getattr(args,field)<=0:parser.error(f'{field} must be positive and finite')
    if not (1<=args.port<=65535 and 1<=args.evidence_port_base<=65532
            and args.port not in range(args.evidence_port_base,args.evidence_port_base+4)):
        parser.error('valid distinct service/evidence ports required')
    import re
    run_name=args.run_name or 'replay-capture-'+uuid.uuid4().hex[:12]
    if not re.fullmatch(r'replay-capture-[A-Za-z0-9_-]+',run_name):parser.error('safe replay-capture-* run-name required')
    allocation=allocated_gpus();contract=load_contract()
    out=args.work_root/run_name;private=out/'private-trace'
    server_name='native-ep-'+run_name
    destination=(args.export_root or args.work_root/'public-reports')/run_name
    service=[sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--execute','--backend','weave_functional',
        '--kernel','tiled','--decode-graph','disabled','--fixed-plan','1,1,0','--hot-cache',
        '--replay-capture-dir',str(private),'--model-path',str(args.model_path),
        '--work-root',str(args.work_root),'--run-name',server_name,
        '--port',str(args.port),'--evidence-port-base',str(args.evidence_port_base)]
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','service_command':service,
            'load_command':load_command(contract['image'],args.work_root,args.model_path,args.dataset,run_name,args.port),
            'required_core_run_dir':str(args.core_run_dir),'private_trace_directory':str(private),
            'used_for_formal_performance':False},indent=2))
        return 0
    if args.reuse_capture_dir is not None:
        try:args.reuse_capture_dir.relative_to(args.work_root)
        except ValueError:parser.error('reused private capture must remain under work-root')
    out.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
         'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':contract['image'],
         'RUNTIME_PYTHON':contract['python_executable']}
    report={'schema_version':1,'status':'running','stage':'validation','used_for_formal_performance':False,
        'started_utc':datetime.now(timezone.utc).isoformat(),'host_gpu_allocation':allocation.csv,
        'core_run_dir':str(args.core_run_dir),'private_trace_directory':str(private),
        'source_sha256':kernel_source_sha256(),'capture_source_sha256':capture_source_sha256()}
    server=OwnedChild();load=OwnedChild();code=1
    def save():atomic_json(out/'capture-run-private.json',report)
    def cancel(signum,frame):
        for child in (server,load):child.requested=signum
        if not any(child.protected for child in (server,load)):raise InterruptedError(f'job signal {signum}')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        save();dataset=validate_dataset(args.dataset);report['dataset']=dataset
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            capture_output=True,text=True,timeout=30,check=True).stdout.strip()
        gate,checkpoint=validate_core_evidence(args.core_run_dir,args.model_path,report['source_sha256'],{'resolved_image_id':image})
        tokenizer=tokenizer_manifest(args.model_path);report['tokenizer_manifest']=tokenizer
        current_identity={'kernel_source_sha256':report['source_sha256'],
            'checkpoint_content_sha256':gate['checkpoint_content_sha256'],'dataset_sha256':dataset['sha256'],
            'capture_source_sha256':report['capture_source_sha256'],'image_id':image,
            'tokenizer_manifest_sha256':tokenizer['sha256']}
        report['core_gate']=gate
        if args.reuse_capture_dir is not None:
            summary=validate_reused_capture(args.reuse_capture_dir,current_identity)
            identity=summary['identity'];report['identity']=identity
            report.update(stage='validated_reuse',private_trace_directory=str(args.reuse_capture_dir))
        else:
            identity={**current_identity,'capture_id':str(uuid.uuid4())}
            validate_identity(identity);report['identity']=identity
            service+=['--checkpoint-content-sha256',identity['checkpoint_content_sha256'],
                      '--replay-dataset-sha256',identity['dataset_sha256'],
                      '--replay-tokenizer-manifest-sha256',identity['tokenizer_manifest_sha256'],
                      '--replay-capture-id',identity['capture_id']]
            bench=load_command(image,args.work_root,args.model_path,args.dataset,run_name,args.port)
            report.update(stage='starting_service',service_command=service,load_command=bench);save()
            assert_ports_free([args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4)))
            with (out/'server-launcher-private.log').open('w') as server_log,(out/'load-manager-private.log').open('w') as load_log:
                process=server.spawn(service,args.work_root/server_name/'container_lifecycle.json',env=env,
                    stdout=server_log,stderr=subprocess.STDOUT)
                url=f'http://127.0.0.1:{args.port}';wait_ready(url,timeout=args.setup_timeout,process=process)
                runtime=json.loads((args.work_root/server_name/'runtime.json').read_text())
                if (runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=image
                        or runtime.get('kernel_source_sha256')!=identity['kernel_source_sha256']
                        or runtime.get('host_gpu_allocation')!=allocation.csv):
                    raise ValueError('runtime differs from validated source/image/allocation')
                if (len(runtime.get('devices',[]))!=4 or any(d.get('capability')!=[9,0] or 'H100' not in d.get('name','')
                        for d in runtime['devices'])):raise ValueError('capture requires four H100 devices')
                if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed')
                if tokenizer_manifest_sha256(args.model_path)!=identity['tokenizer_manifest_sha256']:
                    raise ValueError('tokenizer/config/encoding assets changed during service setup')
                validate_service(coverage(args.evidence_port_base),identity)
                capture_status(args.evidence_port_base,identity)
                for rank in range(4):http_json(f'http://127.0.0.1:{args.evidence_port_base+rank}/replay-capture/arm',{})
                load_lifecycle=out/'load-lifecycle.json'
                managed=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',bench[bench.index('--name')+1],
                    '--evidence',str(load_lifecycle),'--log',str(out/'benchmark-private.log'),'--']+bench
                loader=load.spawn(managed,load_lifecycle,stdout=load_log,stderr=subprocess.STDOUT)
                report['stage']='capturing_real_decode';save();deadline=time.monotonic()+args.capture_timeout
                while True:
                    values=capture_status(args.evidence_port_base,identity)
                    if all(v['coverage']['complete'] for v in values):break
                    if process.poll() is not None:raise RuntimeError('service exited before complete real trace')
                    if loader.poll() is not None:raise RuntimeError('benchmark exited before complete real trace; inspect private log')
                    if time.monotonic()>=deadline:raise TimeoutError('real trace capture deadline; no synthetic substitute')
                    time.sleep(1.)
                if loader.poll() not in (None,0):raise RuntimeError('benchmark failed during real trace capture')
                report['stage']='abort_and_drain';save()
                load.stop();load.check_cancelled();server.check_cancelled()
                http_json(url+'/abort_request',{'abort_all':True},timeout=30)
                report['idle_flush']=flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,
                                                           max_wait=args.drain_timeout)
                report['stage']='persisting_private_tensors';save()
                manifests=[http_json(f'http://127.0.0.1:{args.evidence_port_base+rank}/replay-capture/persist',{},
                                     timeout=args.persist_timeout) for rank in range(4)]
                summary=aggregate_manifests(manifests,identity)
                atomic_json(private/'capture_manifest.json',summary)
                validate_capture_directory(private,identity)
        if validate_dataset(args.dataset)!=dataset:raise ValueError('dataset changed during capture')
        if checkpoint_stats(args.model_path)!=checkpoint['verified_stats']:raise ValueError('checkpoint changed during capture')
        if tokenizer_manifest_sha256(args.model_path)!=identity['tokenizer_manifest_sha256']:
            raise ValueError('tokenizer/config/encoding assets changed during capture/reuse')
        if kernel_source_sha256()!=identity['kernel_source_sha256'] or capture_source_sha256()!=identity['capture_source_sha256']:
            raise ValueError('source changed during capture')
        load.check_cancelled();server.check_cancelled()
        # aggregate_manifests enforces an exact recursive metadata schema.
        # No logs, tensors, input IDs, routing IDs, or benchmark payloads leave work.
        public=aggregate_manifests(summary['ranks'],identity)
        atomic_json(destination/'capture_manifest.json',public)
        report.update(status='capture_complete',stage='complete',public_manifest=str(destination/'capture_manifest.json'))
        code=0
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={name:pool.submit(child.stop) for name,child in (('load',load),('server',server))}
            for name,future in pending.items():
                try:future.result()
                except BaseException as exc:report.update(status='failed',**{name+'_cleanup_error':str(exc)});code=1
        report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'private_trace_directory':report['private_trace_directory'],
        'public_manifest':report.get('public_manifest'),'private_report':str(out/'capture-run-private.json'),
        'used_for_formal_performance':False}),flush=True)
    return code


if __name__=='__main__':
    if sys.argv[1:2]==['--installed-benchmark']:sys.exit(installed_benchmark(sys.argv[2:]))
    sys.exit(main())
