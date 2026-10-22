"""Graph-stable native down/peer-tail executor; GPU source is built offline.

A single IPC workspace is shared by ordered MoE invocations. The GPU sequence and
layer header protect graph replay, including alternating batch graphs. Host code
runs at capture/warmup only; there is no per-token routing readback or recapture.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

from .native_tail_patch import load_down_op, verify_sources, source_pins
from .native_tail_transport import NativeTailTransport
from .native_tail_cost import (METHOD,POLICY,Plan,candidates,encode_surface,load_profile,
                               MAX_PLANS,ANCHORS,ROW_SIZE)

ROOT=Path(__file__).resolve().parents[2]


def deployment_identity(torch,rank,package_root):
    from .native_cost_runtime import deployment_identity as base_identity
    from .native_tail_runtime import native_tail_source_sha256
    value=base_identity(torch,rank,package_root)
    value.pop('cost_source_sha256',None)
    value['tail_source_sha256']=native_tail_source_sha256()
    value['upstream_sha256']=source_pins()
    import importlib.metadata
    distribution=importlib.metadata.distribution('apache-tvm-ffi')
    value['tvm_ffi']=dict(version=distribution.version,files={str(name):hashlib.sha256(distribution.locate_file(name).read_bytes()).hexdigest()
        for name in distribution.files or () if str(name).endswith(('.so','.h','.hpp'))})
    value['method']=METHOD;value['policy']=POLICY
    return value


@dataclass
class Binding:
    layer:int
    batch:int
    group:str
    geometry:dict
    resources:dict
    module:object
    plans:list
    pair:object
    output:object
    state:object
    groups:object
    stats:object
    visits:object
    trace:object
    events:object
    mode:object
    plan_tensor:object
    table:object
    anchors:object
    pair_storage:object
    output_storage:object
    empty_visits:object
    empty_trace:object
    empty_events:object
    down:object=None
    shared:object=None
    owners:list=field(default_factory=list)
    rank_tables:list=field(default_factory=list)
    diagnostic:bool=False


class NativeTailExecutor:
    def __init__(self,torch,rank,package_root,group,*,max_batch=32,cache_root,directory,mode,repeats=7,validation=False):
        import torch.distributed as dist
        self.torch,self.dist,self.rank,self.group=torch,dist,rank,group
        self.package_root=Path(package_root);self.cache_root=Path(cache_root);self.directory=Path(directory)
        self.mode,self.max_batch,self.repeats=mode,int(max_batch),int(repeats)
        self.validation=bool(validation)
        if mode not in ('static','calibration','measured') or self.max_batch<1:raise ValueError('invalid explicit tail mode/capacity')
        self.pins=verify_sources(package_root)
        self.identity=deployment_identity(torch,rank,package_root)
        self.identities=[None]*4;dist.all_gather_object(self.identities,self.identity,group=group)
        self.contexts={};self.modules={};self.stats={};self.profiles={};self.tables={}
        self.layer_groups={i:'g'+str(i) for i in range(43)};self.groups={};self.closed=False
        if mode=='measured':
            for r in range(4):
                self.profiles[str(r)]=load_profile(self.directory/f'native-tail-rank{r}.json',self.identities[r])
            self.configure_groups(self.profiles['0']['groups'],None)
        self.profile=dict(schema_version=1,method=METHOD,policy=POLICY,numerical_policy='record_only',
                          status='in_progress',identity=self.identity,groups=self.groups,tables={})
        self.down_op=load_down_op(package_root,self.cache_root/'down')
        self.transport=NativeTailTransport(group,self.max_batch,4096)
        self.config=self.transport.native_config()

    def configure_groups(self,groups,mapping=None):
        if self.profiles and any(p['groups']!=groups for p in self.profiles.values()):
            raise ValueError('native tail calibrated model class mapping mismatch')
        self.groups=groups
        self.layer_groups=(dict(mapping) if mapping is not None else
                           {layer:name for name,entry in groups.items() for layer in entry['layers']})
        if hasattr(self,'profile'):self.profile['groups']=groups

    def _compile(self,geometry):
        torch=self.torch
        from torch.utils.cpp_extension import load
        from .native_tail_runtime import native_tail_source_sha256
        key=tuple(geometry[k] for k in ('threads','m_blocks','n_blocks','k_blocks','m8','scratch_bytes'))
        if key in self.modules:return self.modules[key]
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('native tail must compile and query occupancy before capture')
        from tvm_ffi import libinfo
        ffi_includes=[libinfo.find_include_path(),libinfo.find_dlpack_include_path()]
        ffi_library=Path(libinfo.find_libtvm_ffi())
        flags=['-O3','-lineinfo','-std=c++20','-Xptxas=-v']
        for name,k in [('THREADS','threads'),('M','m_blocks'),('N','n_blocks'),('K','k_blocks'),('M8','m8')]:
            flags.append(f'-DNT_{name}={int(geometry[k])}')
        digest=hashlib.sha256((native_tail_source_sha256()+json.dumps(key)+str(torch.__version__)).encode()).hexdigest()
        directory=self.cache_root/('executor-'+digest);directory.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a')
        module=load(name='weave_native_tail_'+digest[:20],sources=[str(ROOT/'experimental/native_tail/executor.cu')],
                    build_directory=str(directory),extra_cuda_cflags=flags,extra_cflags=['-O3','-std=c++20'],
                    extra_ldflags=[str(ffi_library),'-Wl,-rpath,'+str(ffi_library.parent)],
                    extra_include_paths=[str(self.down_op.helper_header.parent),str(ROOT/'experimental/native_tail'),*self.down_op.include_dirs,*ffi_includes],verbose=True)
        resources=dict(module.resources(geometry['scratch_bytes']))
        self.modules[key]=(module,resources)
        return module,resources

    def _guarded(self,shape):
        t=self.torch;n=1
        for v in shape:n*=v
        storage=t.full((n+64,),123.,dtype=t.bfloat16,device=f'cuda:{self.rank}')
        return storage,storage[32:-32].view(shape)

    def prepare_inputs(self,down,shared,layer):
        t=self.torch;g=down.geometry;batch=int(g.tokens);key=(layer,batch)
        if not 0<batch<=self.max_batch:raise ValueError('tail batch exceeds allocated IPC capacity')
        if down.activated_pairs.dtype!=t.bfloat16 or down.w2_scales.dtype!=t.float8_e8m0fnu:
            raise ValueError('native BF16 activation and E8M0 scale required')
        if key not in self.contexts:
            if t.cuda.is_current_stream_capturing():raise RuntimeError('native tail binding missing before capture')
            geometry=self.down_op.prepare(g.pair_rows,g.n,g.k,g.block_size_m,self.rank)
            module,resources=self._compile(geometry)
            peers=[None]*4;self.dist.all_gather_object(peers,resources,group=self.group)
            workers=min(r['workers'] for r in peers)
            plans=candidates(workers,g.n,geometry['n_blocks']*16)
            group=self.layer_groups[layer]
            pair_storage,pair=self._guarded((g.pair_rows,g.n));output_storage,output=self._guarded((batch,g.n))
            tensor=lambda shape,dtype:t.zeros(shape,dtype=dtype,device=f'cuda:{self.rank}')
            self.stats.setdefault(layer,tensor(64,t.int64))
            plan_tensor=t.tensor([[p.communication_workers,p.chunks,p.steal_tiles,int(p.serial)] for p in plans],dtype=t.int32,device=f'cuda:{self.rank}')
            max_tasks=g.group_capacity*(g.n//(geometry['n_blocks']*16))
            # Detailed task buffers exist only in independent validation or startup
            # calibration. Production keeps only empty handles and aggregate stats.
            diag=self.mode!='measured' or self.validation
            ctx=Binding(layer,batch,group,geometry,dict(resources,workers=workers),module,plans,pair,output,
                tensor(resources['state_bytes'],t.uint8),tensor(g.group_capacity+(g.pair_rows+31)//32,t.int32),self.stats[layer],
                tensor(max_tasks if diag else 0,t.int32),tensor((workers,10) if diag else 0,t.int64),
                tensor((max_tasks+20,4) if diag else 0,t.int64),
                t.tensor([1,0,0,0],dtype=t.int32,device=f'cuda:{self.rank}'),plan_tensor,
                tensor((4,len(plans),ANCHORS,ROW_SIZE),t.float64),tensor((4,len(plans)),t.int32),
                pair_storage,output_storage,tensor(0,t.int32),tensor(0,t.int64),tensor(0,t.int64))
            ctx.down,ctx.shared=down,shared
            self.contexts[key]=ctx
            if self.mode=='measured':
                table_key=f'{group}:{batch}'
                tables=[self.profiles[str(r)]['tables'].get(table_key) for r in range(4)]
                if any(x is None for x in tables):raise ValueError('missing actual tail shape calibration; run scripts/run_native_tail.py --execute')
                self._upload(ctx,tables);self.set_search(layer,batch)
            elif (group,batch) in self.tables:
                self._upload(ctx,self.tables[group,batch]);self.set_search(layer,batch)
        ctx=self.contexts[key]
        if (ctx.pair.shape!=(g.pair_rows,g.n) or ctx.groups.numel()<g.group_capacity+(g.pair_rows+31)//32):
            raise ValueError('native grouping allocation changed after tail binding')
        ctx.down,ctx.shared=down,shared
        if t.cuda.is_current_stream_capturing():ctx.owners.append((down,shared))
        return ctx

    def __call__(self,down,shared,*,layer_id,routed_scaling_factor,output_width):
        if self.closed:raise RuntimeError('native tail executor is closed')
        if output_width!=4096:raise ValueError('native tail expects actual Flash hidden width4096')
        ctx=self.prepare_inputs(down,shared,layer_id);g=down.geometry
        # Source-bound front marker: front.py reaches this call only after exactly
        # one native gate/up plus activation. Its CUDA work is part of every replay.
        ctx.module.mark_front(ctx.stats,ctx.state,ctx.mode)
        ctx.module.run(down.activated_pairs,down.w2,down.w2_scales,down.sorted_token_ids,
            down.local_expert_ids,down.num_tokens_post_padded,down.topk_weights,shared,
            ctx.pair,ctx.output,ctx.state,ctx.groups,ctx.stats,
            ctx.visits if ctx.diagnostic else ctx.empty_visits,ctx.trace if ctx.diagnostic else ctx.empty_trace,
            ctx.events if ctx.diagnostic else ctx.empty_events,ctx.mode,ctx.plan_tensor,ctx.table,ctx.anchors,
            self.config['addresses'],self.rank,self.max_batch,self.config['session'],layer_id,g.topk,
            ctx.resources['scratch_bytes'],ctx.resources['workers'],float(routed_scaling_factor),30.)
        return ctx.output

    def set_fixed(self,layer,batch,index,diagnostic=False,delay_cycles=0):
        ctx=self.contexts[layer,batch]
        if self.torch.cuda.is_current_stream_capturing():raise RuntimeError('configure validation mode before capture')
        if not 0<=index<len(ctx.plans) or delay_cycles<0 or delay_cycles>=2**31:raise ValueError('invalid fixed plan/delay')
        if diagnostic and not ctx.visits.numel():raise ValueError('formal engine has no per-task diagnostic storage')
        ctx.diagnostic=bool(diagnostic)
        ctx.mode.copy_(self.torch.tensor([1,index,int(diagnostic),delay_cycles],device=ctx.mode.device,dtype=self.torch.int32))

    def set_search(self,layer,batch,diagnostic=False):
        ctx=self.contexts[layer,batch]
        if not ctx.rank_tables:raise ValueError('measured tail table required; run scripts/run_native_tail.py --execute')
        if diagnostic and not ctx.visits.numel():raise ValueError('formal engine cannot enable task diagnostics')
        ctx.diagnostic=bool(diagnostic)
        ctx.mode.copy_(self.torch.tensor([0,0,int(diagnostic),0],device=ctx.mode.device,dtype=self.torch.int32))

    def _upload(self,ctx,tables):
        if len(tables)!=4:raise ValueError('four rank measured curves required')
        for r,table in enumerate(tables):
            if [Plan(**p) for p in table['plans']]!=ctx.plans or table['total_rows']!=ctx.batch*6:
                raise ValueError('calibrated plans or routed geometry differ')
            if r==self.rank and table.get('executor_resources') and table['executor_resources']!=ctx.resources:
                raise ValueError('compiled executor resources differ from calibration')
        data=[encode_surface(table['surface'],ctx.plans,ctx.batch*6) for table in tables]
        ctx.table.copy_(self.torch.tensor(data,dtype=self.torch.float64,device=ctx.table.device))
        ctx.anchors.copy_(self.torch.tensor([[len(a) for a in table['surface']] for table in tables],dtype=self.torch.int32,device=ctx.table.device))
        ctx.rank_tables=tables

    def install_surfaces(self,group,batch,rank_tables):
        self.tables[group,batch]=rank_tables
        for (layer,b),ctx in self.contexts.items():
            if ctx.group==group and b==batch:self._upload(ctx,rank_tables)

    def snapshot_context(self,layer,batch,private=False):
        t=self.torch;t.cuda.synchronize(self.rank);ctx=self.contexts[layer,batch]
        result=dict(ctx.module.snapshot(ctx.state))
        result['protocol']=dict(self.transport.ws.snapshot(ctx.plans[result['selected_index']].chunks))
        result['canary_ok']=all(bool((storage[:32]==123.).all() and (storage[-32:]==123.).all()) for storage in (ctx.pair_storage,ctx.output_storage))
        p=result['protocol']
        result['protocol_valid']=result['canary_ok'] and p.get('abort_code')==0 and p.get('nonfinite')==0 and p.get('acknowledged') is True and p.get('canary_ok') is True
        if ctx.diagnostic:result['protocol_valid'] &= len(p.get('tasks',[]))==p.get('expected_tasks') and all(v==1 for v in p.get('tasks',[]))
        result['resources']=ctx.resources;result['geometry']=ctx.geometry
        if private:
            result.update(visits=ctx.visits.cpu().tolist(),trace=ctx.trace.cpu().tolist(),events=ctx.events.cpu().tolist())
        return result

    def snapshot(self):
        self.torch.cuda.synchronize(self.rank)
        rows=[]
        for layer,state in sorted(self.stats.items()):
            v=state.cpu().tolist()
            rows.append(dict(layer=layer,tail_calls=v[0],gate_up_calls=v[1],committed_calls=v[34],
                plan_search_calls=v[38],empty_calls=v[36],histogram=v[8:30],published_chunks=v[30],
                down_tiles=v[31],stolen_tiles=v[32],remote_read_bytes=v[33]))
        profile=self.profiles.get(str(self.rank),self.profile)
        return dict(schema_version=1,rank=self.rank,mode=self.mode,identity=self.identity,
            source_sha256=self.identity['tail_source_sha256'],profile_sha256=profile.get('profile_sha256'),
            profiles_sha256={r:p.get('profile_sha256') for r,p in self.profiles.items()},
            calibration_complete=profile.get('status')=='complete',rows=rows)

    def close(self):
        if self.closed:return
        # Caller resets all owned graphs before this collective IPC teardown.
        with self.torch.cuda.device(self.rank):
            self.torch.cuda.synchronize(self.rank);self.transport.close();self.closed=True
