#!/usr/bin/env python3
"""Owned offline paper-mechanism probes, calibration, real weights and GPQA."""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_c32_comparison import gate_command,serving_source_manifest
from scripts.run_gpqa_c32 import wait_graph_gate,_safe
from scripts.run_model_acceptance import OwnedChild
from src.weave.calibration import atomic_json
from src.weave.gpu_allocation import allocated_gpus
from src.weave.hopper_persistent import source_manifest
from src.weave.runtime_contract import load_contract


def stage_command(image,work,model,name,allocation,stage,checkpoint):
    command=gate_command(image,work,model,name,allocation,'hopper')
    command[command.index('--name')+1]=f'weave-paper-{stage}-{name}'
    first=command.index('/workspace/tests/gpu/check_hopper_serving.py')
    command[first:]=['/workspace/tests/gpu/check_hopper_persistent.py','--stage',stage,
        '--output-dir',f'/work/{name}/{stage}','--model-path','/model',
        '--checkpoint-content-sha256',checkpoint]
    if stage=='checkpoint':command+=['--profile-dir',f'/work/{name}/calibrate']
    command[command.index('--output-dir')+1]=f'/work/{name}/{stage}'
    return command


def export(run,destination):
    names=['paper-experiment.json','paper-source-manifest.json','serving-source-manifest.json','paper-proof-check.json']
    for stage in ('smoke','calibrate','checkpoint'):
        names += [f'{stage}/runtime.json',f'{stage}/container-lifecycle.json']
        names += [f'{stage}/{prefix}-rank{rank}.json' for prefix in (f'paper-{stage}','paper-profile-bank','graph-cleanup') for rank in range(4)]
    destination.mkdir(parents=True,exist_ok=False);files={}
    for name in names:
        path=run/name
        if path.is_symlink():raise ValueError('paper export symlink forbidden')
        if path.is_file():
            raw=path.read_bytes();_safe(json.loads(raw));target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o664)
            files[name]=hashlib.sha256(raw).hexdigest()
    atomic_json(destination/'manifest.json',{'schema_version':1,'raw_content_exported':False,'files':files})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work-root','records','prefill-evidence-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-path',type=Path,default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--stage-timeout',type=float,default=14400.)
    parser.add_argument('--execute',action='store_true');a=parser.parse_args(argv)
    for key in ('work_root','records','prefill_evidence_dir','model_path'):
        path=getattr(a,key)
        if not path.is_absolute():parser.error(key+' must be absolute')
        setattr(a,key,path.resolve())
    if not math.isfinite(a.stage_timeout) or a.stage_timeout<=0:parser.error('positive stage timeout required')
    if not a.execute:
        print(json.dumps(dict(status='preview_not_executed',method_identity='hopper_persistent_paper_v1',
            complete_weave_claim=False,stages=['smoke','calibrate','checkpoint','native_graph_1200s','paper_graph_1200s'],
            runtime_image=load_contract()['image'],gpu_allocation='GPUQ CUDA_VISIBLE_DEVICES',
            numerical_policy='record_only',profile_buckets=[8,32,128,512,2048],stage_timeout=a.stage_timeout),indent=2));return 0
    from src.weave.gpqa_c32 import validate_prefill_bundle,load_gpqa_workload
    from src.weave.paper_evidence import validate_paper_bundle
    allocation=allocated_gpus();name='paper-reproduction-'+uuid.uuid4().hex[:12]
    out=a.work_root/name;out.mkdir(parents=True,exist_ok=False);destination=a.work_root/'public-reports'/name
    owner=OwnedChild();previous={s:signal.signal(s,owner.signal) for s in (signal.SIGINT,signal.SIGTERM)}
    report=dict(status='running',method_identity='hopper_persistent_paper_v1',method_status='paper_candidate',
        complete_weave_claim=False,numerical_policy='record_only',gpuq_job_id=os.environ.get('GPUQ_JOB_ID'),
        host_gpu_allocation=allocation.csv,stage='validating',started_utc=datetime.now(timezone.utc).isoformat())
    def save():atomic_json(out/'paper-experiment.json',report)
    code=1
    try:
        save();load_gpqa_workload(a.records,model_path=a.model_path)
        contract=load_contract();image=subprocess.run(['docker','image','inspect',contract['image'],'--format','{{.Id}}'],check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        # Previous proof attests checkpoint/image only; the new routed backend
        # must pass its own complete smoke/calibration/checkpoint evidence.
        prior=validate_prefill_bundle(a.prefill_evidence_dir,a.model_path,image);identity=prior['identity']
        identity['image_id']=image;report['identity']=identity
        manifest=source_manifest();atomic_json(out/'paper-source-manifest.json',manifest)
        serving=serving_source_manifest();atomic_json(out/'serving-source-manifest.json',serving)
        for stage in ('smoke','calibrate','checkpoint'):
            owner.check_cancelled();folder=out/stage;folder.mkdir();report['stage']=stage;save()
            command=stage_command(image,a.work_root,a.model_path,name,allocation.csv,stage,identity['checkpoint_content_sha256'])
            lifecycle=folder/'container-lifecycle.json'
            manager=[sys.executable,str(ROOT/'scripts/manage_container.py'),'--name',command[command.index('--name')+1],
                '--evidence',str(lifecycle),'--log',str(folder/'private.log'),'--']+command
            child=owner.spawn(manager,lifecycle,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            wait_graph_gate(child,folder,gate_timeout=a.stage_timeout,cleanup_timeout=120.,cancel_check=owner.check_cancelled)
            owner.stop();owner.check_cancelled()
        proof=validate_paper_bundle(out,identity);atomic_json(out/'paper-proof-check.json',proof)
        report.update(stage='service_comparison',method_status='kernel_mechanisms_verified');save()
        from scripts.run_gpqa_c32 import main as run_service
        stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code=run_service(['--work-root',str(a.work_root),'--records',str(a.records),
                '--prefill-evidence-dir',str(a.prefill_evidence_dir),'--model-path',str(a.model_path),
                '--paper-proof-dir',str(out),'--decode-graph','full','--setup-timeout','14400','--execute'])
        report['service_result']=json.loads(stdout.getvalue().strip().splitlines()[-1])
        if code!=0:raise RuntimeError('paper_service_comparison_failed')
        if source_manifest()!=manifest or serving_source_manifest()!=serving:raise ValueError('source changed during reproduction')
        report.update(status='completed',stage='complete',method_status='paper_service_execution_verified')
    except BaseException as exc:
        code=1;report.update(status='failed',error_type=type(exc).__name__)
        (out/'error-private.log').write_text(str(exc)+'\n')
    finally:
        for s in previous:signal.signal(s,signal.SIG_IGN)
        try:owner.stop()
        except BaseException as exc:code=1;report.update(status='failed',cleanup_error_type=type(exc).__name__)
        report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        try:export(out,destination)
        finally:
            for s,handler in previous.items():signal.signal(s,handler)
    print(json.dumps(dict(status=report['status'],work_directory=str(out),export_directory=str(destination),service_result=report.get('service_result'))))
    return code

if __name__=='__main__':raise SystemExit(main())
