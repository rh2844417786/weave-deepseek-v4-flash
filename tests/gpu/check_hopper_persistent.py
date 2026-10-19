"""Offline four-rank verification/calibration of the paper-mechanism backend."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.weave.hopper_evidence import atomic_report
from src.weave.hopper_persistent import IMPLEMENTATION,MECHANISM_FIELDS
from tests.gpu import check_hopper_serving as common


def inputs(local,h,experts,pattern,variant,device):
    total=local*4;generator=torch.Generator().manual_seed(89117+variant)
    x=(torch.randn((total,h),generator=generator)*.125).bfloat16().to(device)
    rows=[];per=experts//4
    for t in range(total):
        origin=t//local
        if pattern=='local':base=origin*per;row=[base+(t*7+j)%per for j in range(6)]
        elif pattern=='remote':base=((origin+1)%4)*per;row=[base+(t*7+j)%per for j in range(6)]
        elif pattern=='skew':row=[(j+variant)%per for j in range(6)]
        else:row=[(t*7+j*17+variant)%experts for j in range(6)]
        rows.append(row)
    ids=torch.tensor(rows,device=device,dtype=torch.int64)
    routing=torch.full((total,6),1/6,device=device,dtype=torch.float32)
    return x,ids,routing


def call(runtime,values,bound,*,plan=None,profile=None,mode=None,layer=0):
    n=len(values[0])//4;start=runtime.rank*n
    # Any hidden Python routed collective is a defect. Setup/reference Gloo and
    # NCCL operations occur outside this forward boundary.
    original=dist.all_reduce
    def forbidden(*a,**k):raise AssertionError('routed forward called a collective')
    dist.all_reduce=forbidden
    try:return runtime.run(*(v[start:start+n] for v in values),bound,profile=profile,
        fixed_plan=plan,calibration_mode=mode,trace_layer=layer,clamp=10.)
    finally:dist.all_reduce=original


def check_call(runtime,values,*,dynamic):
    torch.cuda.synchronize()
    last=runtime.ws.mechanism_last.cpu().tolist();record=dict(zip(MECHANISM_FIELDS,last))
    smids=runtime.ws.worker_smids.cpu().tolist();resources=dict(runtime.ws.resources())
    workers=torch.cuda.get_device_properties(runtime.device).multi_processor_count
    common.require(resources['active_blocks_per_sm']==1 and resources['grid_blocks']==workers,'one CTA per SM required')
    common.require(len(smids)==workers and len(set(smids))==workers and min(smids)>=0,'actual unique SM residency required')
    common.require(record['calls']==1 and record['dynamic_calls']==int(dynamic),'actual schedule mode differs')
    rows=values[1].cpu().tolist();n=len(rows)//4;k=record['k_sum'];payload=0;tasks=0;remote=0;unique=0;buckets={}
    for token,ids in enumerate(rows):
        selected=[e for e in ids if e//(runtime.experts//4)==runtime.rank]
        if not selected:continue
        chunk=payload*k//record['dispatch_payloads'];payload+=1
        for expert in selected:
            tasks+=1;buckets[expert,chunk]=buckets.get((expert,chunk),0)+1
            if token//n!=runtime.rank:remote+=1
        if token//n!=runtime.rank:unique+=1
    groups=sum((count+63)//64 for count in buckets.values())
    expected=dict(tasks=tasks,dispatch_payloads=payload,remote_pairs=remote,remote_unique=unique,groups=groups,
                  computed_tiles=groups*((runtime.h+63)//64+(runtime.i+63)//64),rows=n)
    common.require(all(record[key]==value for key,value in expected.items()),'routing/group/tile conservation differs')
    common.require(record['stolen_tiles']<=record['computed_tiles'],'stolen count exceeds work')
    return dict(mechanisms=record,unique_worker_smids=True,resources=resources,routed_collectives=0)


def measured(runtime,values,bound,plan,mode):
    call(runtime,values,bound,plan=plan,mode=mode);torch.cuda.synchronize()
    times=[]
    for _ in range(3):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();call(runtime,values,bound,plan=plan,mode=mode);end.record();end.synchronize()
        times.append(float(begin.elapsed_time(end)))
    evidence=check_call(runtime,values,dynamic=False)
    common.require(all(math.isfinite(t) and t>0 for t in times),'positive measured timings required')
    return statistics.median(times),evidence['mechanisms']


def calibration(runtime,bound,buckets,checkpoint):
    workers=torch.cuda.get_device_properties(runtime.device).multi_processor_count
    cs=sorted(set([1,min(8,workers-1),min(16,workers-1),min(32,workers-1)]))
    measured_rows=[];profiles=[]
    for limit in buckets:
        candidates=[]
        local=inputs(limit,runtime.h,runtime.experts,'local',0,f'cuda:{runtime.rank}')
        remote=inputs(limit,runtime.h,runtime.experts,'remote',0,f'cuda:{runtime.rank}')
        for c in cs:
            for k in (1,2,4,8):
                compute,work=common.agreed(f'calibrate_{limit}_{c}_{k}_compute',lambda:measured(runtime,local,bound,(c,k,0),'compute'))
                communication,traffic=common.agreed(f'calibrate_{limit}_{c}_{k}_communication',lambda:measured(runtime,remote,bound,(c,k,0),'communication'))
                flops=work['groups']*64*6*runtime.h*runtime.i
                size=traffic['remote_unique']*runtime.h*2+traffic['remote_pairs']*runtime.h*4
                row=[c,k,flops/(compute/1000),size/(communication/1000),1.,0.]
                candidates.append(row)
                measured_rows.append(dict(max_local_tokens=limit,c=c,k=k,compute_ms=compute,communication_ms=communication,
                    padded_flops=flops,communication_bytes=size,candidate=row))
        profiles.append(dict(max_local_tokens=limit,candidates=candidates))
    prop=torch.cuda.get_device_properties(runtime.device)
    bank=dict(schema_version=1,status='measured',implementation=IMPLEMENTATION,numerical_policy='record_only',
        source_sha256=runtime.source_sha256,device_key=f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}',
        hidden_size=runtime.h,intermediate_size=runtime.i,experts=runtime.experts,top_k=6,rank=runtime.rank,
        image_id=os.environ['WEAVE_IMAGE_ID'],checkpoint_content_sha256=checkpoint,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda),
        model_constants={'efficiency':1.,'overlap':0.},profiles=profiles)
    return bank,measured_rows


def references(runtime,values,bound,raw,reference,nccl,label):
    from tests.gpu.check_hopper_candidate import numerical,local_reference
    from src.weave.hopper_reference import expert_reference_fp8
    from tests.gpu.check_checkpoint import expert_reference
    def paper_output():
        actual=call(runtime,values,bound,profile=runtime.profile,layer=runtime.test_layer).clone()
        torch.cuda.synchronize();return actual.cpu()
    local=common.agreed(label+':paper_output',paper_output)
    parts=[None]*4;dist.all_gather_object(parts,local)
    full=common.agreed(label+':assemble_actual',lambda:torch.cat(parts).to(f'cuda:{runtime.rank}'))
    from types import SimpleNamespace
    # The legacy coordination helper uses clamp=7 for its synthetic fixture.
    # Keep its failure agreement but match this real-weight comparison's 10.
    coordinated=SimpleNamespace(device_group=reference.device_group,_slots=reference._slots,
        run=lambda x,ids,weights,handle,_fixture_clamp:reference.run(x,ids,weights,handle,10.))
    expected=[]
    for start in range(0,len(values[0]),32):
        partial,_=common.run_captured(coordinated,tuple(v[start:start+32] for v in values),reference.bound,label+f':reference_{start}')
        all_output=common.agreed(label+f':reference_copy_{start}',lambda:partial.clone())
        dist.all_reduce(all_output,group=nccl);expected.append(all_output)
    result=common.agreed(label+':original_numeric',lambda:{'original_hopper':numerical(full,torch.cat(expected))})
    for name,fn in [('fp8_oracle',expert_reference_fp8),('bf16_oracle',expert_reference)]:
        def prepare_oracle():
            slots=local_reference(*(v[:2] for v in values),raw,runtime.rank,runtime.experts,10.,fn)
            torch.cuda.synchronize();return slots
        slots=common.agreed(label+':'+name+':prepare',prepare_oracle)
        dist.all_reduce(slots,group=nccl)
        def reduce_oracle():
            summed=torch.zeros((2,runtime.h),device=full.device,dtype=torch.float32)
            for slot in range(6):summed.add_(slots[slot::6])
            return numerical(full[:2],summed.bfloat16())
        result[name]=common.agreed(label+':'+name+':numeric',reduce_oracle)
    common.agreed(label+':finite',lambda:common.require(all(r['finite'] for r in result.values()),'nonfinite paper/reference output'))
    return result


def main():
    global torch,dist
    import torch
    import torch.distributed as dist
    from src.weave.hopper_persistent import HopperPersistentRuntime,source_sha256
    from src.weave.hopper_decode import HopperDecodeRuntime
    from src.weave.weights import load_raw_experts
    from src.weave.graph_cleanup import cleanup_graphs
    from types import SimpleNamespace
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','calibrate','checkpoint'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--profile-dir',type=Path);p.add_argument('--checkpoint-content-sha256',required=True)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('gloo',timeout=timedelta(minutes=30));common.torch,common.dist=torch,dist
    from tests.gpu import check_hopper_candidate as probe
    probe.torch=torch
    os.environ['WEAVE_CHECKPOINT_CONTENT_SHA256']=a.checkpoint_content_sha256
    graphs=[];nccl=None;fresh=False
    report=dict(schema_version=1,status='running',implementation=IMPLEMENTATION,method_status='paper_candidate',
        numerical_policy='record_only',raw_content_exported=False,rank=rank,world_size=dist.get_world_size(),stage=a.stage,cases=[])
    def save():atomic_report(a.output_dir/f'paper-{a.stage}-rank{rank}.json',report)
    def progress(state):
        report['cleanup']=state
        if fresh:atomic_report(a.output_dir/f'graph-cleanup-rank{rank}.json',state)
    try:
        def inspect():
            common.require(dist.get_world_size()==4 and torch.cuda.device_count()==4,'four visible GPUs required')
            common.require(torch.cuda.get_device_capability()==(9,0),'Hopper required')
            common.require(not (a.output_dir/f'paper-{a.stage}-rank{rank}.json').exists(),'fresh paper output required')
        common.agreed('inspect',inspect);fresh=True
        from scripts.run_c32_comparison import serving_source_manifest
        def python_source():return hashlib.sha256(json.dumps(serving_source_manifest(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        report['serving_source_sha256']=python_source()
        report.update(source_sha256=source_sha256(),image_id=os.environ['WEAVE_IMAGE_ID'],checkpoint_content_sha256=a.checkpoint_content_sha256)
        small=a.stage=='smoke';h,i,e,cap=(128,128,32,64) if small else (4096,2048,256,2048)
        runtime=common.agreed('allocate',lambda:HopperPersistentRuntime(dist.group.WORLD,cap,h,i,6,e))
        runtime.experts=e;runtime.phase='decode';runtime.test_layer=0
        report['resources']=dict(runtime.ws.resources());report['shape']=dict(hidden=h,intermediate=i,experts=e,capacity=cap)
        if small:raw=common.agreed('weights',lambda:common.synthetic_weights(rank,f'cuda:{rank}'))
        else:raw=common.agreed('weights',lambda:load_raw_experts(a.model_path,0,rank,4,device=torch.device('cuda',rank)))
        bound=common.agreed('bind',lambda:runtime.bind_weights(raw))
        if a.stage in ('smoke','calibrate'):
            bank,measurements=calibration(runtime,bound,[8,32,64] if small else [8,32,128,512,2048],a.checkpoint_content_sha256)
            atomic_report(a.output_dir/f'paper-profile-bank-rank{rank}.json',bank)
            report['calibration']=measurements
            profile_path=a.output_dir
        else:profile_path=a.profile_dir
        runtime.profile=common.agreed('load_profile',lambda:runtime.profile_from_file(profile_path))
        report['profile_bank_sha256']=runtime.profile.sha256
        if a.stage=='smoke':
            workers=report['resources']['grid_blocks']
            for name,plan,local,pattern in [('serial',(1,1,0),8,'spread'),('dispatch_pipeline',(1,4,0),64,'remote'),('pipeline',(workers-2,4,0),64,'remote'),
                    ('stealing',(workers-2,4,1),64,'remote'),('dynamic',None,8,'spread'),('dynamic_skew',None,32,'skew')]:
                values=inputs(local,h,e,pattern,1,f'cuda:{rank}')
                def run():
                    out=call(runtime,values,bound,plan=plan,profile=runtime.profile if plan is None else None)
                    torch.cuda.synchronize();common.require(bool(torch.isfinite(out).all()),'nonfinite output')
                    return check_call(runtime,values,dynamic=plan is None)
                result=common.agreed(name,run);report['cases'].append(dict(case=name,**result));save()
            # One captured exact B32 local-origin call, then changed routing.
            values=inputs(8,h,e,'spread',2,f'cuda:{rank}');call(runtime,values,bound,profile=runtime.profile);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph();graphs.append(graph)
            with torch.cuda.graph(graph):out=call(runtime,values,bound,profile=runtime.profile)
            for variant in (3,4):
                changed=inputs(8,h,e,'spread',variant,f'cuda:{rank}')
                for old,new in zip(values,changed):old.copy_(new)
                graph.replay();torch.cuda.synchronize()
                result=common.agreed('graph_replay',lambda:check_call(runtime,values,dynamic=True))
                common.require(bool(torch.isfinite(out).all()),'nonfinite graph output')
                report['cases'].append(dict(case=f'graph_{variant}',**result))
        elif a.stage=='checkpoint':
            nccl=dist.new_group(backend='nccl',timeout=timedelta(minutes=5))
            peer=SimpleNamespace(torch=torch,device=rank,rank=rank,ep=4,h=h,i=i,topk=6)
            reference=common.agreed('reference',lambda:HopperDecodeRuntime(peer,nccl,experts=e))
            for layer,local in [(layer,8) for layer in range(43)]+[(0,32)]:
                runtime.test_layer=layer
                runtime.phase='decode' if local==8 else 'prefill'
                raw=common.agreed(f'weights_{layer}',lambda:load_raw_experts(a.model_path,layer,rank,4,device=torch.device('cuda',rank)))
                bound=common.agreed(f'bind_{layer}',lambda:runtime.bind_weights(raw))
                reference.bound=common.agreed(f'reference_bind_{layer}',lambda:reference.bind_weights(raw))
                values=inputs(local,h,e,'spread' if local==8 else 'skew',layer,f'cuda:{rank}')
                numeric=references(runtime,values,bound,raw,reference,nccl,f'layer{layer}')
                evidence=common.agreed('checkpoint_evidence',lambda:check_call(runtime,values,dynamic=True))
                report['cases'].append(dict(case='checkpoint',layer=layer,phase=runtime.phase,batch=local*4,numerical=numeric,numerical_coverage={'original_hopper':{'rows':local*4,'scope':'full_global_batch'},'fp8_oracle':{'rows':2,'global_rows':[0,1]},'bf16_oracle':{'rows':2,'global_rows':[0,1]}},**evidence));save()
        report.update(status='gpu_checks_passed_cleanup_pending',gpu_checks_status='passed',source_sha256_after=source_sha256(),serving_source_sha256_after=python_source())
        common.require(report['serving_source_sha256']==report['serving_source_sha256_after'],'Python source changed')
        common.require(report['source_sha256_after']==report['source_sha256'],'source changed');common.agreed('save',save)
    except Exception as exc:
        common.private_exception('paper_gate',exc);report.update(status='failed',error_type=type(exc).__name__)
        if fresh:save()
    finally:
        try:cleanup_graphs(graphs,torch,dist,nccl,progress)
        except Exception as exc:
            common.private_exception('paper_cleanup',exc);report.update(status='failed',cleanup_error_type=type(exc).__name__)
        else:
            if report['status']=='gpu_checks_passed_cleanup_pending':report['status']='passed'
        if fresh:save()
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
