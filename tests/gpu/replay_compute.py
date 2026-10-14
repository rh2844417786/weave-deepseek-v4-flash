"""EP4 frozen real-trace P0/P1 controls. torchrun inside the existing image only."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.weave.calibration import atomic_json
from src.weave.execution_profile import TraceConfig
from src.weave.ops import PeerRuntime,kernel_source_sha256,kernel_source_manifest,extension
from src.weave.replay_benchmark import ReplayStore,VARIANTS,point_key
from src.weave.replay_trace import load_rank_trace,validate_capture_directory
from src.weave.staged_plan import build_staged_groups
from src.weave.weights import load_raw_experts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--identity-file',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=3)
    parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--stage-repetitions',type=int,default=3)
    parser.add_argument('--budget-seconds',type=float,default=7200.)
    parser.add_argument('--kernel-timeout',type=float,default=120.)
    parser.add_argument('--invocation-id',required=True)
    args=parser.parse_args()
    import torch
    import torch.distributed as dist
    from tests.gpu.check_checkpoint import expert_reference
    if not 1<=args.warmups<=20 or not 2<=args.repetitions<=100 or not 1<=args.stage_repetitions<=10:
        raise ValueError('bounded positive warmup/repetition counts required')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    dist.init_process_group('gloo',timeout=timedelta(seconds=max(1800,args.kernel_timeout*4)))
    if dist.get_world_size()!=4:raise ValueError('four H100 ranks required')
    device=torch.device('cuda',rank);stopped={'signal':None};start=None;store=None
    def remember(number,frame):stopped['signal']=number
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,remember)
    progress=args.output_dir/f'progress-rank{rank}.json'
    stage='initializing'
    def publish(**extra):
        atomic_json(progress,{'rank':rank,'stage':stage,'invocation_id':args.invocation_id,'measurement_started_monotonic':start,
            'budget_seconds':args.budget_seconds,'completed_points':len(store.keys()) if store else 0,
            'signal':stopped['signal'],**extra})
    def local_action(function):
        result=None;error=None
        try:result=function()
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        errors=[None]*4;dist.all_gather_object(errors,error)
        if any(errors):raise RuntimeError('rank action failed: '+str(errors))
        return result
    def should_stop():
        flags=[None]*4
        dist.all_gather_object(flags,stopped['signal'] is not None or (start is not None and time.monotonic()-start>=args.budget_seconds))
        return any(flags)
    def metrics(actual,expected):
        if not torch.isfinite(actual).all().item():raise ValueError('nonfinite replay output')
        error=actual.float()-expected.float()
        rel=float((error.norm()/expected.float().norm().clamp_min(1e-12)).item())
        maximum=float(error.abs().max().item())
        if rel>.025 or not torch.isclose(actual.float(),expected.float(),rtol=.03,atol=.02).all().item():
            raise ValueError(f'replay numerical mismatch relative_l2={rel},max_abs={maximum}')
        return {'status':'passed','relative_l2':rel,'max_abs':maximum}
    try:
        args.output_dir.mkdir(parents=True,exist_ok=True);publish()
        identity_base=json.loads(args.identity_file.read_text())
        if identity_base['source_sha256']!=kernel_source_sha256():raise ValueError('worker CUDA source identity differs')
        manifest=local_action(lambda:validate_capture_directory(args.capture_dir,identity_base['capture_identity']))
        payload=local_action(lambda:load_rank_trace(args.capture_dir,rank,manifest['identity']))
        canonical=payload if rank==0 else local_action(lambda:load_rank_trace(args.capture_dir,0,manifest['identity']))
        # All ranks execute the same collective count, including rank0's no-op load.
        if rank==0:local_action(lambda:None)
        def replica_check():
            if any(not torch.equal(payload[k],canonical[k]) for k in ('x','ids','routing')):
                raise ValueError('captured TP replicas differ; cannot use replicated-input staged control')
        local_action(replica_check)
        prop=torch.cuda.get_device_properties(rank)
        if (prop.major,prop.minor,prop.multi_processor_count)!=(9,0,132):raise ValueError('H100 SM90/132 required')
        identity={**identity_base,'rank':rank,'device_key':f'{prop.name};sm90;{prop.multi_processor_count}',
            'host_gpu_allocation':os.environ.get('WEAVE_HOST_ALLOCATION'),
            'warmups':args.warmups,'repetitions':args.repetitions,'stage_repetitions':args.stage_repetitions,
            'kernel_timeout_seconds':args.kernel_timeout,'torch_version':str(torch.__version__),'cuda':torch.version.cuda}
        store=local_action(lambda:ReplayStore(args.output_dir/f'replay-rank{rank}.json',identity))
        sets=[None]*4;dist.all_gather_object(sets,store.keys())
        local_action(lambda:store.reconcile(set.intersection(*sets)))
        uncached=PeerRuntime(None,2048,4096,2048,6,256,kernel='tiled',cache_views=False)
        cached=PeerRuntime(None,2048,4096,2048,6,256,kernel='tiled',cache_views=True)
        uncached.phase=cached.phase='decode'
        uncached.enable_execution_profile(TraceConfig(stride=16,capacity=8,layers=1))
        staged=extension().StagedWorkspace(rank,4,32,4096,2048,256,6)
        resources={'rank':rank,'source_manifest':kernel_source_manifest(),'source_sha256':kernel_source_sha256(),
            'persistent':uncached.ws.resources(),'staged':staged.resources(),'used_for_formal_performance':False}
        local_action(lambda:atomic_json(args.output_dir/f'resources-rank{rank}.json',resources))
        clamp=float(json.loads((args.model_path/'config.json').read_text())['swiglu_limit'])
        stamps=[time.monotonic() if rank==0 else None];dist.broadcast_object_list(stamps,src=0);start=stamps[0]
        total_expected=43*len(manifest['config']['sample_sequences'])*len(VARIANTS)
        stage='replaying';publish()
        exhausted=False
        for layer in range(43):
            if should_stop():exhausted=True;break
            if all(f'{layer}:{sample}:{variant}' in store.keys() for sample in range(2) for variant in VARIANTS):continue
            raw=local_action(lambda:load_raw_experts(args.model_path,layer,rank,4,device=device))
            bound=local_action(lambda:cached.bind_weights(raw))
            local_action(lambda:staged.bind_weights(list(raw)))
            for sample in range(2):
                if should_stop():exhausted=True;break
                def prepare_trace():
                    x=canonical['x'][sample,layer].to(device)
                    ids=canonical['ids'][sample,layer].to(device)
                    routing=canonical['routing'][sample,layer].to(device)
                    ids_cpu=canonical['ids'][sample,layer].tolist()
                    plan=build_staged_groups(ids_cpu,rank=rank,ep=4,experts=256,topk=6)
                    meta=torch.tensor(plan.group_meta,dtype=torch.int32,device=device)
                    slots=torch.tensor(plan.group_slots,dtype=torch.int32,device=device)
                    count=torch.tensor([plan.group_count],dtype=torch.int32,device=device)
                    staged.bind_inputs(x,ids,routing,meta,slots,count);torch.cuda.synchronize()
                    return x,ids,routing,ids_cpu,plan
                x,ids,routing,ids_cpu,plan=local_action(prepare_trace)
                local_x,local_ids,local_routing=x[rank*8:(rank+1)*8],ids[rank*8:(rank+1)*8],routing[rank*8:(rank+1)*8]
                def oracle():
                    products=torch.zeros((32,6,4096),device=device,dtype=torch.float32)
                    for expert in sorted({e for row in ids_cpu for e in row if e//64==rank}):
                        positions=[(t,s) for t,row in enumerate(ids_cpu) for s,e in enumerate(row) if e==expert]
                        token=torch.tensor([t for t,s in positions],device=device)
                        slot=torch.tensor([s for t,s in positions],device=device)
                        out=expert_reference(x[token],raw,expert%64,clamp)
                        products[token,slot]=(out.bfloat16()*routing[token,slot].bfloat16()[:,None]).float()
                    partial=torch.zeros((32,4096),device=device)
                    for slot in range(6):partial+=products[:,slot]
                    return partial,products.cpu()
                expected_local,products_cpu=local_action(oracle)
                products=[None]*4;dist.all_gather_object(products,products_cpu)
                combined=sum(products)
                expected_global=torch.zeros((32,4096),dtype=torch.float32)
                for slot in range(6):expected_global+=combined[:,slot]
                expected_origin=expected_global.bfloat16()[rank*8:(rank+1)*8].to(device)
                del products,products_cpu,combined
                workload={'pairs':plan.local_pairs,'groups':plan.group_count,
                    'group_row_histogram':{str(rows):sum(n==rows for _,n in plan.group_meta[:plan.group_count])
                                           for rows in sorted({n for _,n in plan.group_meta[:plan.group_count]})},
                    'unique_payloads':len({slot//6 for row in plan.group_slots[:plan.group_count] for slot in row if slot>=0}),
                    'gate_tiles':plan.group_count*32,'down_tiles':plan.group_count*64}
                owners=[None]*4;dist.all_gather_object(owners,plan.local_pairs)
                if sum(owners)!=192:raise ValueError('EP ownership must cover exactly192 token-expert pairs')
                order=list(VARIANTS);random.Random(1441+layer*2+sample).shuffle(order)
                for variant in order:
                    key=f'{layer}:{sample}:{variant}'
                    if key in store.keys():continue
                    if should_stop():exhausted=True;break
                    publish(layer=layer,sample=sample,variant=variant)
                    is_staged=variant.startswith('staged_')
                    grid=(0 if variant=='staged_direct' else int(variant.split('_')[1])) if is_staged else None
                    def launch():
                        if is_staged:
                            staged.run_gate_up(grid,clamp);staged.run_down(grid);staged.run_reduce()
                            return staged.output
                        runtime=cached if variant=='persistent_bound_no_probe' else uncached
                        weights=bound if runtime is cached else raw
                        return runtime.run(local_x,local_ids,local_routing,weights,fixed_plan=(1,1,0),
                            timeout_seconds=args.kernel_timeout,clamp=clamp,
                            trace_layer=0 if variant=='persistent_raw_probe' else None)
                    dist.barrier()
                    actual=local_action(lambda:(launch(),torch.cuda.synchronize())[0])
                    correct=local_action(lambda:metrics(actual,expected_local if is_staged else expected_origin))
                    if not is_staged:
                        work=uncached.ws.workload if variant!='persistent_bound_no_probe' else cached.ws.workload
                        observed=local_action(lambda:work.cpu().tolist())
                        if observed[4:7]!=[plan.local_pairs,plan.group_count,plan.group_count*96]:
                            raise ValueError('persistent and staged task/group populations differ')
                    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(args.repetitions)]
                    for begin,end in events:begin.record();end.record()
                    local_action(torch.cuda.synchronize)
                    for _ in range(args.warmups):
                        dist.barrier();local_action(lambda:(launch(),torch.cuda.synchronize()))
                    if variant=='persistent_raw_probe':local_action(uncached.reset_execution_profile)
                    gpu_samples=[];host_samples=[]
                    for begin,end in events:
                        dist.barrier()
                        def iteration():
                            begin.record();host_start=time.perf_counter();launch()
                            host_ms=(time.perf_counter()-host_start)*1000;end.record();torch.cuda.synchronize()
                            return float(begin.elapsed_time(end)),host_ms
                        gpu_ms,host_ms=local_action(iteration);gpu_samples.append(gpu_ms);host_samples.append(host_ms)
                    stage_samples={}
                    if is_staged:
                        stage_samples={k:[] for k in ('gate_up','down','reduce')}
                        for _ in range(args.stage_repetitions):
                            ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                            for event in ev:event.record()
                            local_action(torch.cuda.synchronize);dist.barrier()
                            def staged_iteration():
                                ev[0].record();staged.run_gate_up(grid,clamp);ev[1].record()
                                staged.run_down(grid);ev[2].record();staged.run_reduce();ev[3].record();torch.cuda.synchronize()
                                return [float(ev[n].elapsed_time(ev[n+1])) for n in range(3)]
                            for name,value in zip(stage_samples,local_action(staged_iteration)):stage_samples[name].append(value)
                    point={'layer':layer,'sample':sample,'variant':variant,'gpu_ms':gpu_samples,
                        'host_enqueue_ms':host_samples,'correctness':correct,'workload':workload,
                        'stage_gpu_ms':stage_samples,'warmups':args.warmups}
                    if variant=='persistent_raw_probe':
                        probe_identity={'rank':rank,'kernel':'tiled','source_sha256':kernel_source_sha256(),
                            'graph_enabled':False,'profile_bank_sha256':None,'fixed_plan':[1,1,0],
                            'schedule_mode':'explicit_fixed','checkpoint_content_sha256':identity_base['checkpoint_content_sha256'],
                            'device_key':identity['device_key']}
                        point['probe_summary']=local_action(lambda:uncached.execution_profile_snapshot(probe_identity))
                    if should_stop():exhausted=True;break
                    local_action(lambda:store.append(point))
                if exhausted:break
            del bound,raw
            if exhausted:break
        status='partial' if exhausted or len(store.keys())!=total_expected else 'complete'
        store.value['status']=status;local_action(store.save)
        stage=status;publish()
        dist.barrier();dist.destroy_process_group()
        # torchrun would translate a deliberate124 into ChildFailedError.
        # The outer owner derives partial/complete from all four bound reports.
        return 0
    except BaseException as exc:
        stage='failed';publish(error_type=type(exc).__name__,error=str(exc))
        if store is not None:store.value['status']='failed';store.save()
        raise


if __name__=='__main__':sys.exit(main())
