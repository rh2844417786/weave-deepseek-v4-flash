#!/usr/bin/env python3
"""Owned, offline C32 tiled/eager/(1,1,0) phase diagnostic, not a speed comparison."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_weave import validate_core_evidence
from scripts.check_sglang_service import wait_ready,coverage,http_json
from scripts.collect_execution_profile import collect_profiles
from scripts.run_model_acceptance import OwnedChild,assert_ports_free
from src.weave.calibration import atomic_json
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.execution_profile import TraceConfig,profiling_source_sha256
from src.weave.experiment_export import export_run
from src.weave.gpu_allocation import allocated_gpus
from src.weave.ops import kernel_source_sha256
from src.weave.performance import flush_cache_when_idle
from src.weave.runtime_contract import load_contract


def diagnostic_source_sha256():
    names=('scripts/run_concurrency_diagnostic.py','scripts/bench_c32_profile.py','scripts/run_native_ep.py',
           'src/weave/experiment_export.py','tests/gpu/check_forward_profile.py')
    return hashlib.sha256(b''.join(name.encode()+b'\0'+(ROOT/name).read_bytes() for name in names)).hexdigest()


def validate_dataset(path):
    payload=Path(path).read_bytes();counts=Counter();records=0
    for line in payload.decode('utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);convs=row.get('conversations',row.get('conversation',[]))
        if not isinstance(convs,list) or len(convs)<2:raise ValueError('custom dataset needs two-turn records')
        prompt=convs[0].get('content',convs[0].get('value'))
        answer=convs[1].get('content',convs[1].get('value'))
        if not isinstance(prompt,str) or not prompt.strip() or not isinstance(answer,str):
            raise ValueError('custom dataset contains a missing text field')
        counts[hashlib.sha256(prompt.encode()).hexdigest()]+=1;records+=1
    if records!=1800 or len(counts)!=9 or set(counts.values())!={200}:
        raise ValueError('reuse the existing nine-case x200 custom dataset (1800 rows); do not substitute a new workload')
    return {'sha256':hashlib.sha256(payload).hexdigest(),'records':records,'unique_prompts':len(counts),
            'repetitions_per_prompt':200,'format':'sglang_custom_jsonl'}


def load_command(image,work,model,dataset,run_name,port,base):
    return ['docker','run','--rm','--pull=never','--name','weave-c32-load-'+run_name,
            '--network=host','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
            '-e','HF_DATASETS_OFFLINE=1','-e','PYTHONUNBUFFERED=1',
            '-v',f'{ROOT}:/workspace:ro','-v',f'{work}:/work','-v',f'{model}:/model:ro',
            '-v',f'{dataset}:/dataset.jsonl:ro',image,load_contract()['python_executable'],
            '/workspace/scripts/bench_c32_profile.py','--marker',f'/work/{run_name}/load-window.json',
            '--evidence-port-base',str(base),'--','--backend','sglang',
            '--base-url',f'http://127.0.0.1:{port}','--model','/model','--tokenizer','/model',
            '--served-model-name','weave-dsv4-functional','--dataset-name','custom',
            '--dataset-path','/dataset.jsonl','--num-prompts','1800','--sharegpt-output-len','1024',
            '--max-concurrency','32','--request-rate','inf','--temperature','0','--warmup-requests','1',
            '--disable-tqdm','--output-file',f'/work/{run_name}/bench-private.jsonl']


def summarize_profiles(values):
    if len(values)!=4 or {v['rank'] for v in values}!={0,1,2,3}:raise ValueError('four ranks required')
    result={'schema_version':1,'measurement_mode':'instrumented_diagnostic','used_for_formal_performance':False,
            'scope':'decode model forwards; replicated TP rows counted once via rank0; not emitted tokens/s',
            'phase_note':'CTA totals and overlapping phase envelopes are not additive layer latency',
            'ranks':{}}
    for value in values:
        forward=value['forward_profile']['phases']['decode'];layers={}
        for key,layer in value['layers'].items():
            recent=layer['decode']['recent_samples']
            layers[key]={'kernel_span_ms_mean':statistics.mean(s['kernel_span_ns']/1e6 for s in recent),
                'recent_samples':len(recent),'sampled_worker_totals_ns':layer['decode']['worker_totals_ns'],
                'mean_group_fill':statistics.mean(s['workload']['tasks']/(16*s['workload']['groups'])
                    if s['workload']['groups'] else 0. for s in recent),
                'phase_envelopes_ms_mean':{name:statistics.mean(s['phase_envelopes_ns'][name]/1e6 for s in recent)
                    for name in recent[0]['phase_envelopes_ns']} if recent else None}
        indexed={key:{s['sequence']:s for s in layer['decode']['recent_samples']}
                 for key,layer in value['layers'].items()}
        aligned=[]
        for sample in forward['recent_samples']:
            seq=sample['sequence']
            if not all(seq in items for items in indexed.values()):continue
            spans=sum(items[seq]['kernel_span_ns'] for items in indexed.values())/1e6
            aligned.append({**sample,'persistent_layers_ms':spans,
                'outside_persistent_ms':sample['gpu_stream_ms']-spans,
                'residual_note':'includes attention/shared experts/prepare/copies/all-reduce/launch gaps; clocks differ'})
        result['ranks'][str(value['rank'])]={'decode_forward':forward,'layers':layers,
            'layer_population':'all retained decode samples; use layers_batch32 for matched C32 samples',
            'aligned_recent_decode':aligned}
        c32={s['sequence'] for s in aligned if s['valid_rows']==32}
        result['ranks'][str(value['rank'])]['layers_batch32']={key:{
            'samples':len(c32),'kernel_span_ms_mean':statistics.mean(items[seq]['kernel_span_ns']/1e6 for seq in c32),
            'phase_envelopes_ms_mean':{name:statistics.mean(items[seq]['phase_envelopes_ns'][name]/1e6 for seq in c32)
                for name in items[next(iter(c32))]['phase_envelopes_ns']},
            'mean_group_fill':statistics.mean(items[seq]['workload']['tasks']/(16*items[seq]['workload']['groups'])
                if items[seq]['workload']['groups'] else 0. for seq in c32)} for key,items in indexed.items()} if c32 else {}
    rank0=next(v for v in values if v['rank']==0)['forward_profile']['phases']['decode']
    result['model_decode_rows_rank0']=rank0['valid_rows_total']
    result['observed_max_decode_batch']=max((int(k) for k in rank0['batch_histogram']),default=0)
    result['fraction_decode_calls_at_batch32']=rank0['batch_histogram'].get('32',0)/rank0['calls'] if rank0['calls'] else 0.
    return result


def validate_diagnostic_completion(summary,progress,finished_early):
    if finished_early:raise ValueError('load exited before the requested window ended')
    if (any(type(progress.get(k)) is not int or progress[k]<0 for k in ('started','succeeded','failed','cancelled'))
            or progress['started']<32 or progress['failed']
            or sum(progress[k] for k in ('succeeded','failed','cancelled'))>progress['started']):
        raise ValueError('formal request counters show failures or invalid concurrency evidence')
    if summary['observed_max_decode_batch']!=32:raise ValueError('diagnostic did not observe decode batch32')
    for rank in summary['ranks'].values():
        c32=[s for s in rank['aligned_recent_decode'] if s['valid_rows']==32]
        if len(c32)<2:raise ValueError('C32 timing requires at least two retained full-forward/layer-aligned samples per rank')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True,help='existing nine SWE-Bench cases x200 custom JSONL')
    parser.add_argument('--core-run-dir',type=Path,required=True,help='passed current-source all43 checkpoint evidence')
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--export-root',type=Path)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--setup-timeout',type=float,default=3600.)
    parser.add_argument('--drain-timeout',type=float,default=120.)
    parser.add_argument('--stride',type=int,default=16)
    parser.add_argument('--capacity',type=int,default=8)
    parser.add_argument('--execute',action='store_true',help='default previews commands without Docker or GPU work')
    args=parser.parse_args(argv)
    for field in ('work_root','dataset','core_run_dir','model_path','export_root'):
        path=getattr(args,field)
        if path is not None:
            if not path.is_absolute():parser.error(f'{field} must be absolute')
            setattr(args,field,path.resolve())
    for name,limit in (('seconds',600.),('setup_timeout',7200.),('drain_timeout',1200.)):
        v=getattr(args,name)
        if not math.isfinite(v) or not 0<v<=limit:parser.error(f'{name} must be finite in (0,{limit}]')
    if not (1<=args.port<=65535 and 1<=args.evidence_port_base<=65532
            and args.port not in range(args.evidence_port_base,args.evidence_port_base+4)):
        parser.error('valid distinct service/evidence ports required')
    config=TraceConfig(stride=args.stride,capacity=args.capacity);allocation=allocated_gpus();contract=load_contract()
    run_name='c32-diagnostic-'+uuid.uuid4().hex[:12];out=args.work_root/run_name
    server_name='native-ep-'+run_name
    destination=(args.export_root or args.work_root/'public-reports')/run_name
    service=[sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--execute','--backend','weave_functional',
        '--kernel','tiled','--decode-graph','disabled','--fixed-plan','1,1,0','--execution-profile',
        '--execution-profile-fixed-plan','--execution-profile-stride',str(config.stride),
        '--execution-profile-capacity',str(config.capacity),'--request-timing',
        '--model-path',str(args.model_path),'--work-root',str(args.work_root),'--run-name',server_name,
        '--port',str(args.port),'--evidence-port-base',str(args.evidence_port_base)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation.csv,'WEAVE_MAX_RUNNING_REQUESTS':'32',
         'WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32','IMAGE':contract['image'],
         'RUNTIME_PYTHON':contract['python_executable']}
    if not args.execute:
        print(json.dumps({'status':'preview_not_executed','service_command':service,
            'checkpoint_identity':'validated from core before execution',
            'service_overrides':{key:env[key] for key in ('WEAVE_MAX_RUNNING_REQUESTS','WEAVE_CUDA_GRAPH_MAX_BS_DECODE')},
            'load_command':load_command(contract['image'],args.work_root,args.model_path,args.dataset,run_name,args.port,args.evidence_port_base)},indent=2))
        return 0
    args.work_root.mkdir(parents=True,exist_ok=True);out.mkdir()
    if destination.exists():raise FileExistsError('fresh export directory required')
    report={'schema_version':1,'status':'running','scope':'c32_fixed_eager_diagnostic',
        'measurement_mode':'instrumented_diagnostic','used_for_formal_performance':False,'performance':'diagnostic_only',
        'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':kernel_source_sha256(),
        'profiling_source_sha256':profiling_source_sha256(),'kernel':'tiled','fixed_plan':[1,1,0],
        'diagnostic_source_sha256':diagnostic_source_sha256(),
        'graph_enabled':False,'host_gpu_allocation':allocation.csv,'gpuq_job_id':os.environ.get('GPUQ_JOB_ID'),
        'core_run_dir':str(args.core_run_dir),'stage':'validating','raw_content_exported':False,
        'requested_load_seconds':args.seconds,'sampling_window':'after benchmark warmup reset through idle after abort/drain',
        'throughput_note':'model-forward rows and sampled device time are not measured emitted token throughput'}
    server=OwnedChild();load=OwnedChild();code=1
    def save():atomic_json(out/'experiment.json',report)
    def cancel(signum,frame):
        for child in (server,load):child.requested=signum
        if not any(child.protected for child in (server,load)):
            raise InterruptedError(f'job signal {signum}')
    previous={sig:signal.signal(sig,cancel) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        save();report['dataset']=validate_dataset(args.dataset)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            capture_output=True,text=True,timeout=30,check=True).stdout.strip()
        gate,identity=validate_core_evidence(args.core_run_dir,args.model_path,report['source_sha256'],{'resolved_image_id':image})
        report['core_gate']=gate;report['resolved_image_id']=image
        report['stage']='forward_event_smoke';save()
        smoke_name='weave-c32-smoke-'+run_name
        smoke=['docker','run','--rm','--pull=never','--name',smoke_name,'--network=none',
            '--gpus',allocation.docker_argument,'-v',f'{ROOT}:/workspace:ro','-v',f'{args.work_root}:/work',
            image,contract['python_executable'],'/workspace/tests/gpu/check_forward_profile.py',
            '--output',f'/work/{run_name}/forward-profile-smoke.json']
        smoke_args=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',smoke_name,
            '--evidence',str(out/'smoke-lifecycle.json'),'--log',str(out/'smoke-private.log'),'--']+smoke
        smoke_process=load.spawn(smoke_args,out/'smoke-lifecycle.json',stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        try:
            if smoke_process.wait(timeout=120)!=0:raise RuntimeError('forward GPU event smoke failed')
        finally:load.stop()
        load.check_cancelled();server.check_cancelled()
        smoke_report=json.loads((out/'forward-profile-smoke.json').read_text())
        if smoke_report.get('status')!='passed' or smoke_report.get('profiling_source_sha256')!=report['profiling_source_sha256']:
            raise ValueError('forward GPU smoke source mismatch')
        service+=['--checkpoint-content-sha256',gate['checkpoint_content_sha256']]
        bench=load_command(image,args.work_root,args.model_path,args.dataset,run_name,args.port,args.evidence_port_base)
        report['service_command']=service;report['load_command']=bench;report['stage']='starting_service';save()
        assert_ports_free([args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4)))
        with (out/'server-launcher-private.log').open('w') as server_log,(out/'load-manager-private.log').open('w') as load_log:
            process=server.spawn(service,args.work_root/server_name/'container_lifecycle.json',env=env,
                                 stdout=server_log,stderr=subprocess.STDOUT)
            url=f'http://127.0.0.1:{args.port}';wait_ready(url,timeout=args.setup_timeout,process=process)
            runtime=json.loads((args.work_root/server_name/'runtime.json').read_text())
            if (runtime['runtime_status']!='passed' or runtime['resolved_image_id']!=image
                    or runtime['kernel_source_sha256']!=report['source_sha256']
                    or runtime.get('host_gpu_allocation')!=allocation.csv):raise ValueError('runtime differs from validated contract')
            if checkpoint_stats(args.model_path)!=identity['verified_stats']:raise ValueError('checkpoint changed')
            before=coverage(args.evidence_port_base)
            for rank,item in enumerate(before):
                if (item.get('rank')!=rank or item.get('kernel')!='tiled' or item.get('graph_enabled') is not False
                        or item.get('schedule_mode')!='explicit_fixed' or item.get('fixed_plan')!=[1,1,0]
                        or item.get('profile_bank_sha256') is not None
                        or item.get('source_sha256')!=gate['source_sha256']
                        or item.get('profile_checkpoint_content_sha256')!=gate['checkpoint_content_sha256']
                        or item.get('execution_profile_configuration')!={'stride':config.stride,'capacity':config.capacity,'layers':43}
                        or set(item.get('layers',{}))!={str(i) for i in range(43)}):
                    raise ValueError('service is not the requested all43 fixed/eager diagnostic')
            report['stage']='benchmark_setup';save()
            load_lifecycle=out/'load-lifecycle.json'
            load_args=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',bench[bench.index('--name')+1],
                '--evidence',str(load_lifecycle),'--log',str(out/'benchmark-private.log'),'--']+bench
            loader=load.spawn(load_args,load_lifecycle,stdout=load_log,stderr=subprocess.STDOUT)
            marker=out/'load-window.json';setup_deadline=time.monotonic()+args.setup_timeout
            while not marker.exists():
                if loader.poll() is not None:raise RuntimeError('benchmark exited before the first formal request; inspect private benchmark log')
                if process.poll() is not None:raise RuntimeError('service exited during benchmark setup')
                if time.monotonic()>=setup_deadline:raise TimeoutError('benchmark setup/warmup timeout')
                time.sleep(.1)
            start=json.loads(marker.read_text())['started_monotonic']
            if not 0<=time.monotonic()-start<=args.setup_timeout:raise ValueError('invalid load marker clock')
            report.update(stage='measuring',load_started_monotonic=start);save()
            deadline=start+args.seconds
            while time.monotonic()<deadline and loader.poll() is None:
                if process.poll() is not None:raise RuntimeError('service exited during measurement')
                progress_path=out/'load-progress.json'
                if progress_path.exists() and json.loads(progress_path.read_text()).get('failed',0):break
                time.sleep(min(.2,max(0,deadline-time.monotonic())))
            report['load_stop_requested_monotonic']=time.monotonic()
            report['load_window_seconds']=report['load_stop_requested_monotonic']-start
            report['load_finished_early']=loader.poll() is not None
            if loader.poll() not in (None,0):raise RuntimeError('benchmark failed before window end; inspect private log')
            load.stop();load.check_cancelled();server.check_cancelled()
            report['request_counts']=json.loads((out/'load-progress.json').read_text())
            report['stage']='draining_owned_service';save()
            http_json(url+'/abort_request',{'abort_all':True},timeout=30)
            report['idle_flush']=flush_cache_when_idle(url,deadline=time.monotonic()+args.drain_timeout,
                                                       max_wait=args.drain_timeout)
            report['idle_monotonic']=time.monotonic();report['stage']='collecting';save()
            devices=runtime['devices']
            if len(devices)!=4 or any(d['capability']!=[9,0] or 'H100' not in d['name'] for d in devices):
                raise ValueError('four H100 devices required')
            bounds={'profiles':[{'rank':rank,'device_key':f'{devices[rank]["name"]};sm90;132'} for rank in range(4)]}
            report['profile_collection']=collect_profiles(args.evidence_port_base,out/'weave',config,gate,bounds,
                allocation.csv,report['profiling_source_sha256'],fixed_plan=[1,1,0])
            values=[json.loads((out/'weave'/f'execution-profile-rank{r}.json').read_text()) for r in range(4)]
            summary=summarize_profiles(values);atomic_json(out/'weave'/'diagnostic-summary.json',summary)
            validate_diagnostic_completion(summary,report['request_counts'],report['load_finished_early'])
            if validate_dataset(args.dataset)!=report['dataset']:raise ValueError('dataset changed during diagnostic')
            if kernel_source_sha256()!=report['source_sha256'] or profiling_source_sha256()!=report['profiling_source_sha256']:
                raise ValueError('source changed during diagnostic')
            if diagnostic_source_sha256()!=report['diagnostic_source_sha256']:raise ValueError('diagnostic driver changed during execution')
            load.check_cancelled();server.check_cancelled()
            report['status']='diagnostic_complete';code=0
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save()
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={name:pool.submit(child.stop) for name,child in (('load',load),('server',server))}
            for name,future in pending.items():
                try:future.result()
                except BaseException as exc:report.update(status='failed',**{name+'_cleanup_error':str(exc)});code=1
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        for label,source in [('launch.json',args.work_root/server_name/'launch.json'),
                             ('runtime.json',args.work_root/server_name/'runtime.json'),
                             ('server-lifecycle.json',args.work_root/server_name/'container_lifecycle.json')]:
            if source.exists():atomic_json(out/label,json.loads(source.read_text()))
        report['export_directory']=str(destination);save()
        export_run(out,destination)
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'export_directory':str(destination),
                      'used_for_formal_performance':False}),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
