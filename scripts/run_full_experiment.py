#!/usr/bin/env python3
"""GPUQ entry: tiled correctness, measured calibration, and matched GPQA groups."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
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
from scripts.check_sglang_service import wait_ready,coverage
from scripts.calibrate_weave import validate_core_evidence
from src.weave.checkpoint_identity import checkpoint_stats
from src.weave.experiment_export import export_run
from src.weave.experiment import (ExperimentLedger,validate_profile_directory,validate_model_evidence,
                                  validate_tiled_snapshot_metadata,prepare_export_directory)
from src.weave.ops import kernel_source_sha256
from src.weave.performance import benchmark_group,compare_groups,_atomic_json,_load_records
from src.weave.request_timing import source_sha256 as timing_source_sha256
from src.weave.runtime_contract import load_contract
from src.weave.service_acceptance import validate_coverage_delta
from src.weave.calibration import measuring_deadline
from src.weave.gpu_allocation import allocated_gpus
from src.weave.execution_profile import add_profile_arguments,profile_options,profile_configuration,TraceConfig,profiling_source_sha256
from scripts.collect_execution_profile import reset_profiles,wait_idle,collect_profiles


def archive_evidence(report,run_dir,destination):
    report['export_directory']=str(destination)
    _atomic_json(Path(run_dir)/'experiment.json',report)
    return export_run(run_dir,destination)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root',type=Path,default=Path(os.environ.get('WORK_ROOT',ROOT/'work')))
    parser.add_argument('--records',type=Path,default=ROOT/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl')
    parser.add_argument('--core-run-dir',type=Path,help='reuse only a passed current-source tiled core')
    parser.add_argument('--calibration-dir',type=Path,help='reuse/resume matching raw event points under work-root')
    parser.add_argument('--model-acceptance-dir',type=Path,help='reuse matching measured-profile full-model evidence')
    parser.add_argument('--calibration-invocations',type=int,default=4,help='each measurement invocation capped at1200s')
    parser.add_argument('--calibration-kernel-timeout-seconds',type=float,default=60.,
                        help='device protocol wait guard, calibration only; does not change1200s budget')
    parser.add_argument('--export-root',type=Path,
                        help='writable public evidence disk; default WORK_ROOT/public-reports')
    parser.add_argument('--setup-timeout',type=float,default=3600.)
    parser.add_argument('--port',type=int,default=30001)
    parser.add_argument('--evidence-port-base',type=int,default=30110)
    add_profile_arguments(parser)
    args=parser.parse_args(argv)
    try: profile_options(args)
    except ValueError as error: parser.error(str(error))
    if not 0<args.calibration_kernel_timeout_seconds<=1200:
        parser.error('calibration kernel timeout must be finite in (0,1200]')
    if args.export_root is not None and not args.export_root.is_absolute():
        parser.error('absolute export-root required')
    return args,parser


def calibration_command(args,core,cal,lifecycle,invocation):
    return [sys.executable,str(ROOT/'scripts/run_calibration_job.py'),'--model-path',str(args.model_path),
        '--work-root',str(args.work_root),'--core-run-dir',str(core),'--output-dir',str(cal),
        '--lifecycle-path',str(lifecycle),'--invocation-id',invocation,
        '--kernel-timeout-seconds',str(args.calibration_kernel_timeout_seconds)]


def main():
    contract=load_contract()
    args,parser=parse_args()
    profiling_options=profile_options(args);profiling_config=profile_configuration(args)
    measurement_mode='instrumented_diagnostic' if args.execution_profile else 'formal_performance'
    try: allocation=allocated_gpus()
    except ValueError as error: parser.error(str(error))
    os.environ['CUDA_VISIBLE_DEVICES']=allocation.csv
    if not args.work_root.is_absolute() or not args.model_path.is_absolute(): parser.error('absolute model/work paths required')
    if not args.records.is_file(): parser.error('prepare pinned448-row Think High IDs offline before submitting')
    _,prepared=_load_records(args.records)
    if not 1<=args.calibration_invocations<=8 or not 0<args.setup_timeout<=7200:
        parser.error('calibration invocations1..8 and setup timeout in(0,7200] required')
    if not 1<=args.port<=65535 or not 1<=args.evidence_port_base<=65532 or args.port in range(args.evidence_port_base,args.evidence_port_base+4):
        parser.error('valid distinct serving/evidence ports required')
    args.work_root=args.work_root.resolve();args.work_root.mkdir(parents=True,exist_ok=True)
    for name in ('core_run_dir','calibration_dir','model_acceptance_dir'):
        path=getattr(args,name)
        if path:
            if not path.is_absolute(): parser.error('reuse paths must be absolute')
            path=path.resolve()
            try: path.relative_to(args.work_root)
            except ValueError: parser.error('all reuse paths must be under work-root')
            setattr(args,name,path)
    run_id=uuid.uuid4().hex[:12]
    destination=prepare_export_directory(args.work_root,f'full-experiment-{run_id}',args.export_root)
    out=args.work_root/f'full-experiment-{run_id}';out.mkdir()
    ledger=ExperimentLedger();child=OwnedChild();benchmark_active=False
    source=kernel_source_sha256()
    report={'schema_version':1,'status':'running','scope':'instrumented_gpqa_diagnostic' if args.execution_profile else 'formal_gpqa_budgeted_comparison',
            'measurement_mode':measurement_mode,'used_for_formal_performance':not args.execution_profile,
            'execution_profile_configuration':profiling_config,
            'profiling_source_sha256':profiling_source_sha256() if args.execution_profile else None,
            'run_id':run_id,'started_utc':datetime.now(timezone.utc).isoformat(),
            'kernel':'tiled','source_sha256':source,'timing_source_sha256':timing_source_sha256(),
            'calibration_kernel_timeout_seconds':args.calibration_kernel_timeout_seconds,
            'steps':ledger.steps,'physical_gpus':list(allocation.ids),'host_gpu_allocation':allocation.csv,
            'acceptance_policy':{'logprob':'record_only','top_logprob_support':'record_only'},
            'performance':'not_run','raw_content_exported':False,
            'dataset_record_sha256':prepared['record_sha256']}
    path=out/'experiment.json'
    def checkpoint(): _atomic_json(path,report)
    def ensure_source():
        if kernel_source_sha256()!=source or timing_source_sha256()!=report['timing_source_sha256']:
            raise ValueError('experiment source changed during execution')
        if args.execution_profile and profiling_source_sha256()!=report['profiling_source_sha256']:
            raise ValueError('execution profiling source changed during execution')
    def interrupted(number,frame):
        if benchmark_active and child.requested is None:
            child.requested=number
            raise KeyboardInterrupt(f'GPUQ signal{number}')
        child.signal(number,frame)
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    def execute(command,lifecycle,label,*,env=None,calibration_marker=None,invocation=None,allowed=(0,)):
        ensure_source();started=time.monotonic()
        with (out/f'{label}.log').open('w') as log:
            process=child.spawn(command,lifecycle,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    child.check_cancelled()
                    deadline=measuring_deadline(calibration_marker,invocation) if calibration_marker else None
                    if deadline is not None and time.monotonic()>=deadline:
                        child.stop();return 124
                    if deadline is None and time.monotonic()-started>args.setup_timeout:
                        raise TimeoutError(f'{label} setup/validation timeout')
                    time.sleep(.2)
                code=process.returncode
                if code not in allowed: raise subprocess.CalledProcessError(code,command)
                return code
            finally: child.stop()
    try:
        checkpoint();print(f'Full experiment evidence: {out}',flush=True)
        image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],
            capture_output=True,text=True,check=True,timeout=30).stdout.strip()
        core=args.core_run_dir or args.work_root/f'cuda-functional.{run_id}'
        report['core_run_dir']=str(core);checkpoint()
        if args.core_run_dir is None:
            env={**os.environ,'CHECKPOINT_TESTS':'1','CHECKPOINT_LAYERS':'all',
                 'MODEL_PATH':str(args.model_path),'WORK_ROOT':str(args.work_root),'CORE_RUN_NAME':core.name}
            execute(['bash',str(ROOT/'scripts/run_cuda_functional_gpuq.sh')],core/'container_lifecycle.json','core',env=env)
        core_gate,identity=validate_core_evidence(core,args.model_path,source,{'resolved_image_id':image})
        ledger.record('core',core_gate);report['core_run_dir']=str(core);checkpoint()
        cal=args.calibration_dir or args.work_root/f'calibration.{run_id}'
        report['calibration_dir']=str(cal);checkpoint()
        for attempt in range(args.calibration_invocations):
            try:
                profiles=validate_profile_directory(cal,core_gate)
                break
            except (OSError,ValueError,KeyError):
                invocation=f'{run_id}-{attempt}'
                command=calibration_command(args,core,cal,out/f'calibration-{attempt}-lifecycle.json',invocation)
                if cal.exists(): command.append('--resume')
                execute(command,out/f'calibration-{attempt}-lifecycle.json',f'calibration-{attempt}',
                    calibration_marker=cal/'progress-rank0.json',invocation=invocation,allowed=(0,124))
        profiles=validate_profile_directory(cal,core_gate)
        ledger.record('calibration',profiles);report['calibration_dir']=str(cal);checkpoint()
        if args.model_acceptance_dir:
            model=args.model_acceptance_dir
        else:
            before=set(args.work_root.glob('model-acceptance-*'))
            command=[sys.executable,str(ROOT/'scripts/run_model_acceptance.py'),'--model-path',str(args.model_path),
                '--work-root',str(args.work_root),'--records',str(args.records),'--core-run-dir',str(core),
                '--kernel','tiled','--profile',str(cal),'--port',str(args.port),
                '--evidence-port-base',str(args.evidence_port_base)]
            controller_lifecycle=out/'model-controller-lifecycle.json'
            command += ['--controller-lifecycle',str(controller_lifecycle)]
            command += profiling_options
            # The driver stops its own separately-sessioned launcher. Give the
            # outer owner one extra second for the nested cleanup handshake.
            child.cleanup_timeout=9
            try: execute(command,controller_lifecycle,'model-acceptance')
            finally: child.cleanup_timeout=8
            created=set(args.work_root.glob('model-acceptance-*'))-before
            if len(created)!=1: raise ValueError('cannot identify unique model acceptance artifact')
            model=created.pop()
        gate=validate_model_evidence(model,core_gate,profiles,profiling_config)
        if gate['record_sha256']!=report['dataset_record_sha256']:
            raise ValueError('prepared GPQA input changed during setup')
        ledger.record('model_acceptance',gate);report['model_acceptance_dir']=str(model);checkpoint()
        report['performance']='running'
        groups={}
        for label,backend in [('native','native'),('weave','weave_functional')]:
            if hashlib.sha256(args.records.read_bytes()).hexdigest()!=report['dataset_record_sha256']:
                raise ValueError('prepared GPQA input changed before formal group')
            if validate_profile_directory(cal,core_gate)!=profiles:
                raise ValueError('measured profile bank changed after full-model validation')
            ensure_source();assert_ports_free([args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4)))
            name=f'native-ep-{run_id}-formal-{label}'
            command=[sys.executable,str(ROOT/'scripts/run_native_ep.py'),'--execute','--backend',backend,
                '--kernel','tiled','--decode-graph','full','--request-timing','--model-path',str(args.model_path),
                '--work-root',str(args.work_root),'--run-name',name,'--port',str(args.port),
                '--evidence-port-base',str(args.evidence_port_base),'--checkpoint-content-sha256',identity['sha256']]
            if backend=='weave_functional': command += ['--profile',str(cal)]+profiling_options
            with (out/f'{label}-launcher.log').open('w') as log:
                process=child.spawn(command,args.work_root/name/'container_lifecycle.json',stdout=log,stderr=subprocess.STDOUT)
                try:
                    url=f'http://127.0.0.1:{args.port}'
                    wait_ready(url,timeout=args.setup_timeout,process=process)
                    ready=time.monotonic()
                    if checkpoint_stats(args.model_path)!=identity['verified_stats']:
                        raise ValueError('checkpoint changed before formal group')
                    runtime=json.loads((args.work_root/name/'runtime.json').read_text())
                    if (runtime['resolved_image_id']!=image or runtime['runtime_status']!='passed'
                            or runtime['kernel_source_sha256']!=source
                            or runtime.get('host_gpu_allocation')!=allocation.csv):
                        raise ValueError('formal runtime/image/source/allocation differs from passed gates')
                    before=coverage(args.evidence_port_base) if backend=='weave_functional' else None
                    if before: validate_tiled_snapshot_metadata(before,core_gate,profiles,profiling_config)
                    if before and args.execution_profile: reset_profiles(args.evidence_port_base)
                    benchmark_active=True
                    try: result=benchmark_group(url,args.records,out/label,ready_monotonic=ready,raw_root=args.work_root,
                                                measurement_mode=measurement_mode)
                    finally: benchmark_active=False
                    if before:
                        if args.execution_profile: wait_idle(url)
                        after=coverage(args.evidence_port_base)
                        validate_tiled_snapshot_metadata(after,core_gate,profiles,profiling_config)
                        result['coverage']=validate_coverage_delta(before,after,require_graph=True)
                        if args.execution_profile:
                            result['execution_profile']=collect_profiles(args.evidence_port_base,out/label,
                                TraceConfig(**profiling_config),core_gate,profiles,allocation.csv,report['profiling_source_sha256'])
                    result['runtime']=runtime;result['server_run_directory']=str(args.work_root/name)
                    _atomic_json(out/label/'summary.json',result)
                    groups[label]=result
                finally: child.stop()
            child.check_cancelled()
            result['cleanup']=json.loads((args.work_root/name/'container_lifecycle.json').read_text())
            if result['cleanup']['cleanup_status'] not in ('absent','stopped'):
                raise ValueError('formal service cleanup not confirmed')
            _atomic_json(out/label/'summary.json',result)
            ledger.record(label+'_formal',result);checkpoint()
        comparison=compare_groups(groups['native'],groups['weave'])
        ledger.record('comparison',comparison)
        _atomic_json(out/'comparison.json',comparison)
        report['status']=('budgeted_diagnostic_complete' if args.execution_profile and ledger.outcome=='budgeted_comparison_complete'
                          else ledger.outcome)
        report['performance']='instrumented_diagnostic' if args.execution_profile else 'measured_budgeted_common_completed_rows'
    except BaseException as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try: child.stop()
        finally:
            for s,handler in previous.items(): signal.signal(s,handler)
            report['finished_utc']=datetime.now(timezone.utc).isoformat();checkpoint()
            try: archive_evidence(report,out,destination)
            except Exception as export_error:
                report['export_error']=f'{type(export_error).__name__}: {export_error}'
                checkpoint()
                if report['status']!='failed': raise
    print(json.dumps({'status':report['status'],'evidence':str(path),'export_directory':str(destination),
                     'measurement_mode':measurement_mode,'used_for_formal_performance':not args.execution_profile,
                     'relative_decode_speed':report['steps']['comparison'].get('relative_decode_speed')}),flush=True)


if __name__=='__main__': main()
