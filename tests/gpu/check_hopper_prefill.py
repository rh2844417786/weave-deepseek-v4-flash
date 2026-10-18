"""Actual four-H100 compact prefill runtime matrix; aggregate public evidence only.

Reference chunk32 loops occur exclusively in this GPU check, outside serving.
CPU oracle snapshots are bounded to two tokens; no tensors enter public JSON.
"""
import argparse
from datetime import timedelta
import gc
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.weave.hopper_evidence import atomic_report
from src.weave.hopper_prefill_evidence import expected_cases
from tests.gpu import check_hopper_serving as smoke

def require(value,message):
    if not value:raise AssertionError(message)

def metadata(ws,ids,rank,experts):
    rows=ids.cpu().tolist();local=experts//4;slots=[[] for _ in range(local)]
    for token,row in enumerate(rows):
        for slot,expert in enumerate(row):
            if expert//local==rank:slots[expert%local].append(token*6+slot)
    counts=[len(x) for x in slots]
    require(ws.group_counts.cpu().tolist()==counts,'group counts differ')
    actual=ws.expert_slots.cpu()
    for expert,items in enumerate(slots):
        require(actual[expert,:len(items)].tolist()==items,'stable expert slots differ')
    tiles=[(expert,base) for expert,items in enumerate(slots) for base in range(0,len(items),64)]
    require(ws.tile_count.cpu().tolist()==[len(tiles)],'tile count differs')
    require(ws.tile_experts[:len(tiles)].cpu().tolist()==[x[0] for x in tiles],'tile experts differ')
    require(ws.tile_row_bases[:len(tiles)].cpu().tolist()==[x[1] for x in tiles],'tile row bases differ')
    return {'local_slots':sum(counts),'active_tiles':len(tiles),'max_expert_rows':max(counts)}

def addresses(runtime):
    return tuple(getattr(runtime.ws,k).data_ptr() for k in ('slot_outputs','output','group_counts','expert_slots','tile_experts','tile_row_bases','tile_count'))

def inputs(batch,variant,experts,hidden,device):
    generator=torch.Generator().manual_seed(73197+variant)
    x=(torch.randn((batch,hidden),generator=generator)*.125).bfloat16().to(device)
    if variant in (1,5):rows=[[slot+experts//4 for slot in range(6)] for _ in range(batch)]
    elif variant==2:rows=[[slot for slot in range(6)] for _ in range(batch)]
    else:rows=[[(token*7+slot*5+variant)%experts for slot in range(6)] for token in range(batch)]
    return x,torch.tensor(rows,device=device,dtype=torch.int64),torch.tensor([[.1,.15,.2,.1,.25,.2]]*batch,device=device)

def metrics(actual,expected):
    from tests.gpu.check_hopper_candidate import numerical
    return numerical(actual,expected)

def evaluate(runtime,old,baseline,raw,bound,old_bound,baseline_bound,values,rank,experts,clamp,label):
    from tests.gpu.check_hopper_candidate import local_reference
    from src.weave.hopper_reference import expert_reference_fp8
    from tests.gpu.check_checkpoint import expert_reference
    x,ids,routing=values;batch=x.shape[0]
    ptr=addresses(runtime);counter=runtime.additional_slot_collectives
    original=dist.all_reduce;entered=False;calls=0;failure=None;output=None;observations={}
    def observed(tensor,*args,**kwargs):
        nonlocal entered,calls
        entered=True;calls+=1
        require(calls==1,'runtime issued more than one prefill slot collective')
        def inspect():
            torch.cuda.synchronize();observations['compute_wall_seconds']=time.monotonic()-started
            require(calls==1 and tensor is runtime._slots[batch] and kwargs.get('group') is runtime.device_group and kwargs.get('op')==dist.ReduceOp.SUM,'wrong prefill collective')
            require(bool(torch.isfinite(tensor).all()),'nonfinite local slots')
            observations.update(metadata(runtime.ws,ids,rank,experts))
            remote=(ids.flatten()//(experts//4)!=rank)
            require(not bool(tensor[remote].count_nonzero()),'slot ownership differs')
            squares=reference_squares=0.;maximum=0.
            # Unchanged FP8 decoder workspace, bounded to its original B<=32.
            for start in range(0,batch,32):
                end=min(batch,start+32);old.run(x[start:end],ids[start:end],routing[start:end],old_bound,clamp)
                expected=old.slot_outputs[:(end-start)*6];actual=tensor[start*6:end*6]
                require(bool(torch.isfinite(expected).all()),'nonfinite old FP8 reference')
                delta=actual-expected;squares+=delta.double().square().sum().item();reference_squares+=expected.double().square().sum().item();maximum=max(maximum,delta.abs().max().item())
            observations['numerical']={'old_fp8':{'finite':True,'relative_l2':(squares/max(reference_squares,1e-300))**.5,'max_abs':maximum}}
            # Only two tokens of each real tensor reach independent Torch oracles.
            count=min(batch,2);actual=tensor[:count*6]
            for name,oracle in (('fp8_oracle',expert_reference_fp8),('bf16_oracle',expert_reference)):
                expected=local_reference(x[:count],ids[:count],routing[:count],raw,rank,experts,clamp,oracle)
                value=metrics(actual,expected);require(value['finite'],'nonfinite oracle difference');observations['numerical'][name]=value
            baseline.run_online(x[:count],ids[:count],routing[:count],baseline_bound,clamp)
            value=metrics(actual,baseline.online_slot_outputs[:count*6]);require(value['finite'],'nonfinite BF16 reference');observations['numerical']['bf16_reference']=value
            observations['oracle_tokens']=count
        smoke.agreed(label+':pre_collective',inspect)
        result=original(tensor,*args,**kwargs)
        # SUM output checked against the exact ordered six-slot BF16 assembly.
        def assembly():
            expected=torch.zeros((batch,x.shape[1]),device=x.device,dtype=torch.float32)
            for slot in range(6):expected.add_(tensor[slot::6])
            expected=expected.bfloat16();chunk=(batch+3)//4
            expected[:rank*chunk].zero_();expected[min((rank+1)*chunk,batch):].zero_()
            observations['_expected']=expected
        smoke.agreed(label+':summed_slots',assembly)
        return result
    dist.all_reduce=observed
    started=time.monotonic()
    try:
        output=runtime.run(x,ids,routing,bound,clamp);torch.cuda.synchronize()
    except Exception as exc:
        smoke.private_exception(label,exc);failure=type(exc).__name__
    finally:dist.all_reduce=original
    if not entered:
        try:smoke.agreed(label+':pre_collective',lambda:require(False,'runtime did not reach collective'))
        except smoke.AgreedFailure:pass
    smoke.agreed(label+':complete',lambda:require(failure is None and calls==1,'runtime failed'))
    def check():
        require(torch.equal(output,observations.pop('_expected')),'ordered origin assembly differs')
        require(bool(torch.isfinite(output).all()),'nonfinite output')
        require(ptr==addresses(runtime),'buffers moved')
        require(runtime.additional_slot_collectives-counter==1,'collective counter differs')
        return dict(observations,metadata_exact=True,finite=True,assembly_exact=True,stable_addresses=True,binding_alive=True,additional_slot_collectives=1)
    return smoke.agreed(label+':check',check)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);a=p.parse_args(argv)
    global torch,dist
    import torch
    import torch.distributed as dist
    from src.weave.hopper_prefill import HopperPrefillRuntime,runtime_source_sha256
    from src.weave.hopper_prefill_candidate import source_sha256
    from src.weave.hopper_candidate import extension,source_sha256 as old_source
    from src.weave.ops import kernel_source_sha256,extension as bf16_extension
    from src.weave.checkpoint_identity import checkpoint_identity,checkpoint_stats
    from src.weave.weights import load_raw_experts
    import json
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('gloo',timeout=timedelta(minutes=120));smoke.torch=torch;smoke.dist=dist
    from tests.gpu import check_hopper_candidate as probe
    probe.torch=torch
    report={'schema_version':1,'rank':rank,'world_size':dist.get_world_size(),'status':'failed','numerical_policy':'record_only','cases':[],'raw_content_exported':False};group=None;fresh=False
    def save():atomic_report(a.output_dir/f'hopper-prefill-rank{rank}.json',report)
    try:
        def inspect():
            require(dist.get_world_size()==4 and dist.get_rank()==rank,'single node four ranks required')
            props=torch.cuda.get_device_properties(rank);require(torch.cuda.device_count()==4 and 'H100' in props.name and (props.major,props.minor)==(9,0),'four H100 required')
            require(not (a.output_dir/f'hopper-prefill-rank{rank}.json').exists(),'fresh prefill report required')
            config=json.loads((a.model_path/'config.json').read_text())
            require((config['num_hidden_layers'],config['hidden_size'],config['moe_intermediate_size'],config['n_routed_experts'],config['num_experts_per_tok'])==(43,4096,2048,256,6),'original checkpoint contract required')
            return config
        config=smoke.agreed('inspect',inspect);fresh=True
        smoke.agreed('pinned_image',lambda:require(isinstance(os.environ.get('WEAVE_IMAGE_ID'),str) and os.environ['WEAVE_IMAGE_ID'].startswith('sha256:'),'resolved existing image required'))
        stats=smoke.agreed('checkpoint_stats',lambda:checkpoint_stats(a.model_path))
        identity=smoke.agreed('checkpoint_content',lambda:checkpoint_identity(a.model_path) if rank==0 else None);identities=[None]*4;dist.all_gather_object(identities,identity)
        report.update(prefill_source_sha256=source_sha256(),prefill_runtime_source_sha256=runtime_source_sha256(),candidate_source_sha256=old_source(),bf16_source_sha256=kernel_source_sha256(),checkpoint_content_sha256=identities[0]['sha256'],image_id=os.environ.get('WEAVE_IMAGE_ID'))
        group=dist.new_group(ranks=list(range(4)),backend='nccl',timeout=timedelta(minutes=10));device=f'cuda:{rank}'
        for real in (False,True):
            hidden,intermediate,experts=(4096,2048,256) if real else (128,128,32)
            peer=SimpleNamespace(torch=torch,device=rank,rank=rank,ep=4,h=hidden,i=intermediate,topk=6)
            runtime=smoke.agreed('create_runtime',lambda:HopperPrefillRuntime(peer,group,experts=experts));old=smoke.agreed('create_old',lambda:extension().FP8Workspace(rank,4,32,hidden,intermediate,experts,6))
            baseline=smoke.agreed('create_bf16',lambda:bf16_extension().StagedWorkspace(rank,4,32,hidden,intermediate,experts,6))
            report.setdefault('resources',{})['real' if real else 'synthetic']={key:value for key,value in runtime.resources.items() if type(value) in (int,float,str,bool)}
            require(runtime.candidate_source_sha256==report['prefill_source_sha256'] and runtime.runtime_source_sha256==report['prefill_runtime_source_sha256'],'loaded runtime differs')
            selected=[case for case in expected_cases() if (case['layer']>=0)==real]
            current_layer=None;raw=bound=old_bound=None
            for index,case in enumerate(selected):
                if current_layer!=case['layer'] or raw is None:
                    raw=smoke.agreed('load_weights',lambda:load_raw_experts(a.model_path,case['layer'],rank,4,device=device) if real else smoke.synthetic_weights(rank,device))
                    bound=smoke.agreed('bind_prefill',lambda:runtime.bind_weights(raw));old_bound=smoke.agreed('bind_old',lambda:old.bind_weights(raw));baseline_bound=smoke.agreed('bind_bf16',lambda:baseline.bind_online_weights(raw));current_layer=case['layer']
                variant=(0 if case['case']=='b1_shrink' else index) if not real else case['layer']
                values=smoke.agreed('inputs',lambda:inputs(case['batch'],variant,experts,hidden,device))
                result=evaluate(runtime,old,baseline,raw,bound,old_bound,baseline_bound,values,rank,experts,float(config['swiglu_limit']) if real else 7.,case['case']+str(case['layer'])+str(case['batch']))
                report['cases'].append(dict(case,**result));smoke.agreed('save_progress',save)
            if not real:
                def lifetime():
                    fresh_raw=smoke.synthetic_weights(rank,device)
                    handle=runtime.bind_weights(fresh_raw)
                    del fresh_raw
                    gc.collect()
                    tiny=inputs(1,0,experts,hidden,device)
                    runtime.ws.run(*tiny,handle,7.);torch.cuda.synchronize()
                    require(bool(torch.isfinite(runtime.ws.slot_outputs[:6]).all()),'binding lifetime execution nonfinite')
                    return {'status':'passed','external_raw_references_dropped':True,'bound_handle_execution_after_gc':True,'local_execution_finite':True}
                report['binding_lifetime']=smoke.agreed('binding_lifetime',lifetime)
            del runtime,old,baseline,raw,bound,old_bound,baseline_bound,values;gc.collect();torch.cuda.empty_cache()
        smoke.agreed('unchanged_checkpoint',lambda:require(checkpoint_stats(a.model_path)==stats,'checkpoint changed'))
        smoke.agreed('unchanged_sources',lambda:require(source_sha256()==report['prefill_source_sha256'] and runtime_source_sha256()==report['prefill_runtime_source_sha256'] and old_source()==report['candidate_source_sha256'] and kernel_source_sha256()==report['bf16_source_sha256'],'source changed'))
        report.update(status='passed',case_count=len(report['cases']));smoke.agreed('save',save);return 0
    except Exception as exc:
        smoke.private_exception('main',exc);report['stop_reason']=str(exc) if isinstance(exc,smoke.AgreedFailure) else 'runtime_failure'
        if fresh:save()
        return 1
    finally:
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()
if __name__=='__main__':sys.exit(main())
