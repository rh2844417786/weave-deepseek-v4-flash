#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include "marlin_down_tile.cuh"
#include "peer_reduce.cuh"
#include "cost_model.h"
namespace cg=cooperative_groups;
namespace weave::native_tail {
struct ExecutorState {
  unsigned cursor,groups,tasks_per_chunk,done[4],finalize_claim[4],finalized,stop;
  unsigned invalid,selected;
  double features[5];
  int chosen[4];
  double prediction;
  TailCurve cache[4*kCostMaxPlans];
  U64 times[4][8]; // diagnostic only: down min/max, finalize, owner, gather
  U64 marker_start,start,planned,finish,planner_begin,planner_end,completion_begin,role_switch,first_ready;
};
struct ExecutorArgs {
  DownTileArgs down;
  TransportConfig transport;
  const __nv_bfloat16* shared;
  float routed_scale;
  int topk,block_rows,tiles,workers,scratch_bytes,plan_count;
  int* groups;
  ExecutorState* state;
  U64* stats; // aggregate per-layer counters, retained in the formal window
  int* visits; // optional validation task bitmap, never retained in serving
  U64* events;
  U64* trace; // optional [workers,10] validation role/SM/time records
  const int* mode; // device [mode:0=search/1=fixed,index,diagnostic]
  const TailPlan* plans;
  const double* table; // [rank,plan,anchor,row]
  const int* anchors;
};
__device__ inline U64 timer_ns(){U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ inline void tick_range(U64*times,U64 begin,U64 end){atomicMin(times,begin);atomicMax(times+1,end);}
__device__ inline void feature_scan(const ExecutorArgs&a){
  auto*s=a.state;int padded=*a.down.num_tokens_padded;
  if(padded<0||padded%a.block_rows||padded>a.down.group_capacity*a.block_rows){s->invalid=1;return;}
  int rows[64]={};int groups=0;unsigned total=0,seen_count=0;
  unsigned* seen=reinterpret_cast<unsigned*>(a.groups+a.down.group_capacity);
  if(a.down.num_experts!=64){s->invalid=5;return;}
  for(int i=0;i<(a.down.pair_rows+31)/32;++i)seen[i]=0;
  for(int g=0;g<padded/a.block_rows;++g){int e=a.down.expert_ids[g];
    if(e< -1||e>=a.down.num_experts||e>=64){s->invalid=2;return;}
    bool padding=false;int count=0;
    for(int j=0;j<a.block_rows;++j){int slot=a.down.sorted_token_ids[g*a.block_rows+j];
      if(slot<0){s->invalid=3;return;}if(slot>=a.down.pair_rows)padding=true;
      else{if(padding){s->invalid=4;return;}unsigned bit=1u<<(slot%32);
        if(seen[slot/32]&bit){s->invalid=6;return;}seen[slot/32]|=bit;++seen_count;++count;}}
    if(e>=0){a.groups[groups++]=g;rows[e]+=count;total+=count;}
  }
  if(seen_count!=unsigned(a.down.pair_rows)){s->invalid=7;return;}
  s->groups=groups;s->features[0]=total;s->features[1]=0;s->features[2]=groups;s->features[3]=0;s->features[4]=0;
  for(int e=0;e<64;++e){s->features[1]+=rows[e]>0;s->features[3]+=double(rows[e])*rows[e];s->features[4]=nt_max(s->features[4],rows[e]);}
}
__device__ inline bool choose_consensus(const ExecutorArgs&a,U64 start){
  auto&cfg=a.transport;auto*local=control(cfg,cfg.rank);auto&box=local->planner;
  U64 epoch=local->sequence+1;
  for(int i=0;i<5;++i)box.workload[i]=a.state->features[i];
  box.workload[5]=a.mode[0];box.workload[6]=a.mode[1];box.workload[7]=cfg.layer;
  sys_store(&box.workload_epoch,epoch);
  if(cfg.rank==0){
    for(int r=0;r<4;++r)while(sys_load(&control(cfg,r)->planner.workload_epoch)<epoch)
      if(protocol_aborted(cfg,start))return false;
    double f[20];for(int r=0;r<4;++r){auto&b=control(cfg,r)->planner;
      if(sys_load(&b.workload_epoch)!=epoch||b.workload[5]!=a.mode[0]||b.workload[7]!=cfg.layer){abort_local(cfg,kIdentity);return false;}
      if(a.mode[0]==1&&b.workload[6]!=a.mode[1]){abort_local(cfg,kIdentity);return false;}
      for(int i=0;i<5;++i)f[r*5+i]=b.workload[i];}
    if(a.mode[0]==0){a.state->prediction=select_tail_plan(a.table,a.anchors,f,a.plans,a.plan_count,a.down.pair_rows,a.state->cache,a.state->chosen);}
    else {if(a.mode[1]<0||a.mode[1]>=a.plan_count){abort_local(cfg,kIdentity);return false;}
      for(int r=0;r<4;++r)a.state->chosen[r]=a.mode[1];a.state->prediction=0;}
    for(int r=0;r<4;++r)box.plan[r]=a.state->chosen[r];
    sys_store(&box.plan_epoch,epoch);
  }
  auto&leader=control(cfg,0)->planner;
  while(sys_load(&leader.plan_epoch)<epoch)if(protocol_aborted(cfg,start))return false;
  if(sys_load(&leader.plan_epoch)!=epoch){abort_local(cfg,kIdentity);return false;}
  for(int r=0;r<4;++r)a.state->chosen[r]=leader.plan[r];
  a.state->selected=a.state->chosen[cfg.rank];
  if(a.state->selected>=unsigned(a.plan_count)){abort_local(cfg,kIdentity);return false;}
  sys_store(&box.plan_ack,epoch);
  // No rank may overwrite its mailbox before the leader has read all features.
  for(int r=0;r<4;++r)while(sys_load(&control(cfg,r)->planner.plan_ack)<epoch)
    if(protocol_aborted(cfg,start))return false;
  return true;
}
__device__ inline bool try_finalize(const ExecutorArgs&a,const TransportConfig&cfg,Invocation inv){
  __shared__ int chunk;
  if(threadIdx.x==0){chunk=-1;for(int c=0;c<cfg.chunks&&chunk<0;++c)
    if(dev_load(&a.state->done[c])==a.state->tasks_per_chunk&&atomicCAS(&a.state->finalize_claim[c],0u,1u)==0)chunk=c;}
  __syncthreads();if(chunk<0)return false;
  const U64 begin=a.mode[2]?timer_ns():0;
  int width=a.down.n/(cfg.chunks*4);size_t n=size_t(cfg.tokens)*width*4;
  auto*pair=reinterpret_cast<__nv_bfloat16*>(a.down.output);
  for(size_t i=threadIdx.x;i<n;i+=blockDim.x){int owner=i/(size_t(cfg.tokens)*width),token=(i/width)%cfg.tokens,x=i%width;
    int channel=(chunk*4+owner)*width+x;float sum=0;
    for(int k=0;k<a.topk;++k)sum+=__bfloat162float(pair[(size_t(token)*a.topk+k)*a.down.n+channel]);
    // Native moe_topk_sum rounds BF16 before outer shared.add_(alpha=scale).
    float routed=__bfloat162float(__float2bfloat16_rn(sum));
    float shared=__bfloat162float(a.shared[size_t(token)*a.down.n+channel]);
    __nv_bfloat16 value=__float2bfloat16_rn(fmaf(a.routed_scale,routed,shared));
    if(!isfinite(__bfloat162float(value)))abort_local(cfg,kNonfinite);
    cfg.layout.partial(cfg.peers[cfg.rank],inv.slot)[payload_index(cfg,chunk,owner,token,x)]=value;
  }
  publish_local_chunk(cfg,inv,chunk);
  if(threadIdx.x==0){cuda::atomic_ref<unsigned,cuda::thread_scope_device>(a.state->finalized).fetch_add(1,cuda::memory_order_acq_rel);
    atomicAdd(a.stats+30,1ULL);if(a.mode[2])tick_range(a.state->times[chunk]+2,begin,timer_ns());}
  __syncthreads();return true;
}
__device__ inline bool do_down(const ExecutorArgs&a,const TransportConfig&cfg,int4*scratch,bool stolen){
  __shared__ unsigned id;
  if(threadIdx.x==0){unsigned total=a.state->tasks_per_chunk*cfg.chunks;
    id=dev_load(&a.state->cursor);while(id<total){unsigned prior=atomicCAS(&a.state->cursor,id,id+1);if(prior==id)break;id=prior;}}
  __syncthreads();unsigned total=a.state->tasks_per_chunk*cfg.chunks;if(id>=total)return false;
  int chunk=id/a.state->tasks_per_chunk,within=id%a.state->tasks_per_chunk;
  int per=a.tiles/cfg.chunks;int group=a.groups[within/per],tile=chunk*per+within%per;
  U64 start=a.mode[2]?timer_ns():0;
  run_marlin_down_tile<NT_THREADS,NT_M,NT_N,NT_K,NT_M8,4>(a.down,group,tile,
    scratch,a.scratch_bytes);
  // Every producer orders its stores; release completion follows CTA rendezvous.
  __threadfence(); __syncthreads();
  if(threadIdx.x==0){
    cuda::atomic_ref<unsigned,cuda::thread_scope_device>(a.state->done[chunk]).fetch_add(1,cuda::memory_order_acq_rel);
    atomicAdd(a.stats+31,1ULL);if(stolen)atomicAdd(a.stats+32,1ULL);
    if(a.visits)atomicAdd(a.visits+group*a.tiles+tile,1);
    if(a.mode[2])tick_range(a.state->times[chunk],start,timer_ns());
    if(a.events){U64*e=a.events+(group*a.tiles+tile)*4;e[0]=blockIdx.x;e[1]=start;e[2]=timer_ns();e[3]=1;}
    if(a.trace){a.trace[blockIdx.x*10+4]++;a.trace[blockIdx.x*10+6]+=stolen;}
  }
  __syncthreads();return true;
}
__device__ inline bool communication_ready(const TransportConfig&cfg,Invocation inv){
  for(int c=0;c<cfg.chunks;++c){bool all=true;for(int r=0;r<4;++r)all&=sys_load(&slot_control(cfg,r,inv)->ready[c])==inv.sequence;
    if(all)return true;}
  return false;
}
__global__ void mark_front_kernel(U64*stats,ExecutorState*state,const int*mode){if(threadIdx.x==0){state->marker_start=mode[2]?timer_ns():0;stats[1]++;}}
__global__ void resident_executor(ExecutorArgs a){
  extern __shared__ int4 scratch[];
  cg::grid_group grid=cg::this_grid();auto*s=a.state;auto cfg=a.transport;cfg.diagnostics=a.mode[2]!=0;
  if(blockIdx.x==0&&threadIdx.x==0){
    s->cursor=s->groups=s->tasks_per_chunk=s->finalized=s->stop=s->invalid=0;s->prediction=0;
    for(int c=0;c<4;++c){s->done[c]=s->finalize_claim[c]=0;for(int j=0;j<8;++j)s->times[c][j]=j%2?0:~0ULL;}
    s->start=a.mode[2]?s->marker_start:0;s->planner_begin=s->planner_end=s->completion_begin=s->role_switch=s->first_ready=0;feature_scan(a);
    s->planner_begin=a.mode[2]?timer_ns():0;
    if(s->invalid)abort_local(cfg,kIdentity);
    else choose_consensus(a,clock64());
    s->planner_end=a.mode[2]?timer_ns():0;
  }
  // Empty remote slots must be zero, as in native EP. No down task writes them.
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(a.down.pair_rows)*a.down.n;i+=gridDim.x*blockDim.x)
    reinterpret_cast<__nv_bfloat16*>(a.down.output)[i]=__float2bfloat16(0);
  if(a.visits)for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.down.group_capacity*a.tiles;i+=gridDim.x*blockDim.x)a.visits[i]=0;
  if(a.events)for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<(a.down.group_capacity*a.tiles+20)*4;i+=gridDim.x*blockDim.x)a.events[i]=0;
  grid.sync();
  if(sys_load(&control(cfg,cfg.rank)->abort_code)){asm("trap;");return;}
  TailPlan plan=a.plans[s->selected];cfg.chunks=plan.chunks;
  if(blockIdx.x==0&&threadIdx.x==0){
    if(plan.comm>=a.workers||a.tiles%plan.chunks||a.down.n%(4*plan.chunks*16*NT_N)||
        (plan.comm<2&&plan.steal)||(plan.serial&&(plan.comm||plan.chunks!=1||plan.steal))){abort_local(cfg,kIdentity);}
    s->tasks_per_chunk=s->groups*(a.tiles/plan.chunks);prepare_invocation(cfg);
    atomicAdd(a.stats,1ULL);atomicAdd(a.stats+8+s->selected,1ULL);
    atomicAdd(a.stats+36,U64(s->groups==0));atomicAdd(a.stats+38,U64(a.mode[0]==0));
    s->planned=a.mode[2]?timer_ns():0;
  }
  grid.sync();Invocation inv=current_invocation(cfg);
  const bool comm_role=blockIdx.x<unsigned(plan.comm);int quota=(comm_role&&blockIdx.x!=0)?plan.steal:0;
  bool entered_comm=false;U64 stolen_end=0;const U64 start=clock64();
  if(a.trace&&threadIdx.x==0){U64*tr=a.trace+blockIdx.x*10;unsigned sm;asm("mov.u32 %0, %%smid;":"=r"(sm));
    for(int j=0;j<10;++j)tr[j]=0;tr[0]=comm_role;tr[1]=sm;tr[2]=timer_ns();}
  __shared__ bool stop,headers,can_comm,ready;
  while(true){
    if(threadIdx.x==0){stop=dev_load(&s->stop)||protocol_aborted(cfg,start);headers=protocol_headers_ready(cfg,inv);
      ready=headers&&communication_ready(cfg,inv);can_comm=(comm_role&&!plan.serial)||dev_load(&s->finalized)==unsigned(cfg.chunks);}
    __syncthreads();if(stop)break;
    if(comm_role&&ready){entered_comm=true;if(a.mode[2]&&threadIdx.x==0)atomicCAS(&s->first_ready,0ULL,timer_ns());if(a.trace&&threadIdx.x==0&&!a.trace[blockIdx.x*10+7])a.trace[blockIdx.x*10+7]=timer_ns();}
    bool worked=false;
    if(headers&&can_comm){U64 begin=a.mode[2]?timer_ns():0;int task=transport_try_work(cfg,inv,a.mode[2]?&s->first_ready:nullptr);
      if(task>=0){worked=true;entered_comm=true;
        if(threadIdx.x==0){
          int c=task<cfg.chunks?task:(task-cfg.chunks)/4;int stage=task<cfg.chunks?4:6;
          if(a.mode[2])tick_range(s->times[c]+stage,begin,timer_ns());
          if(a.events){U64*e=a.events+(a.down.group_capacity*a.tiles+task)*4;e[0]=blockIdx.x;e[1]=begin;e[2]=timer_ns();e[3]=task<cfg.chunks?2:3;}
          if(stolen_end){U64 observed_ready=atomicAdd(&s->first_ready,0ULL);U64 dependency=stolen_end>observed_ready?stolen_end:observed_ready;
            if(begin>dependency)atomicMax(&s->role_switch,begin-dependency);stolen_end=0;}
          if(a.trace){auto*tr=a.trace+blockIdx.x*10;tr[5]++;if(!tr[8])tr[8]=begin?begin:timer_ns();}
          U64 bytes=U64(cfg.tokens)*(a.down.n/(cfg.chunks*4))*2;
          if(task<cfg.chunks)atomicAdd(a.stats+33,bytes*3);
          else if((task-cfg.chunks)%4!=cfg.rank)atomicAdd(a.stats+33,bytes);
        }
      }
    }
    if(!worked&&!comm_role&&(!a.mode[2]||clock64()-start>=U64(a.mode[3]))){worked=try_finalize(a,cfg,inv);if(!worked)worked=do_down(a,cfg,scratch,false);}
    if(!worked&&comm_role&&!entered_comm&&quota>0&&!ready){worked=do_down(a,cfg,scratch,true);if(worked){--quota;if(a.mode[2])stolen_end=timer_ns();}}
    if(threadIdx.x==0){
      if(a.mode[2]&&dev_load(&slot_control(cfg,cfg.rank,inv)->completed)==unsigned(cfg.chunks*5))atomicCAS(&s->completion_begin,0ULL,timer_ns());
      if(headers&&transport_complete(cfg,inv))dev_store(&s->stop,1);
      if(!worked)__nanosleep(64);
    }
    __syncthreads();
  }
  if(a.trace&&threadIdx.x==0)a.trace[blockIdx.x*10+3]=timer_ns();
  grid.sync();
  if(blockIdx.x==0&&threadIdx.x==0){s->finish=a.mode[2]?timer_ns():0;atomicAdd(a.stats+34,1ULL);}
  if(sys_load(&control(cfg,cfg.rank)->abort_code)){asm("trap;");}
}
} // namespace
namespace {
using namespace weave::native_tail;
void check(cudaError_t code){TORCH_CHECK(code==cudaSuccess,cudaGetErrorString(code));}
pybind11::dict resources(int scratch){
  int device;check(cudaGetDevice(&device));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device));
  TORCH_CHECK(prop.cooperativeLaunch,"cooperative launch required; no oversubscribed fallback");
  cudaFuncAttributes initial;check(cudaFuncGetAttributes(&initial,resident_executor));
  scratch=std::min(scratch,int(prop.sharedMemPerBlockOptin-initial.sharedSizeBytes))/16*16;
  constexpr int m=NT_M8?8:16*NT_M;
  constexpr int red=(2*NT_N+1)*16*NT_M,b=4*(8*NT_N*NT_K),bias=2*NT_N;
  constexpr int bred=((red>b?red:b)>((red<b?red:b)+bias))?(red>b?red:b):((red<b?red:b)+bias);
  constexpr int minimum=1024+16*(m+bred+4*(NT_K/2)*NT_N+4*m*(2*NT_K));
  TORCH_CHECK(scratch>=minimum,"final executor scratch cannot hold native asynchronous pipeline");
  check(cudaFuncSetAttribute(resident_executor,cudaFuncAttributeMaxDynamicSharedMemorySize,scratch));
  int active;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,resident_executor,NT_THREADS,scratch));
  cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,resident_executor));
  TORCH_CHECK(active>0,"native tail final kernel has no resident configuration");
  pybind11::dict d;d["workers"]=active*prop.multiProcessorCount;d["active_blocks_per_sm"]=active;
  d["sms"]=prop.multiProcessorCount;d["registers"]=attr.numRegs;d["local_bytes"]=attr.localSizeBytes;
  d["static_shared_bytes"]=attr.sharedSizeBytes;d["scratch_bytes"]=scratch;d["threads"]=NT_THREADS;
  d["state_bytes"]=sizeof(ExecutorState);return d;
}
void launch(torch::Tensor activation,torch::Tensor weight,torch::Tensor scales,torch::Tensor sorted,
  torch::Tensor experts,torch::Tensor padded,torch::Tensor routing,torch::Tensor shared,
  torch::Tensor pair,torch::Tensor output,torch::Tensor state,torch::Tensor groups,
  torch::Tensor stats,torch::Tensor visits,torch::Tensor trace,torch::Tensor events,torch::Tensor mode,
  torch::Tensor plans,torch::Tensor table,torch::Tensor anchors,std::vector<uint64_t> peers,
  int rank,int capacity,uint64_t session,int layer,int topk,int scratch,int workers,double scale,double timeout){
  c10::cuda::CUDAGuard guard(activation.device());
  for(auto&t:{activation,weight,scales,sorted,experts,padded,routing,shared,pair,output,state,groups,stats,visits,trace,events,mode,plans,table,anchors})
    TORCH_CHECK(t.is_cuda()&&t.device()==activation.device()&&t.is_contiguous(),"native tail CUDA contiguous device mismatch");
  int rows=activation.size(0),n=output.size(1),k=activation.size(1),tokens=output.size(0);
  int block=NT_M8?8:16*NT_M,capacity_groups=std::min<int>(experts.numel(),sorted.numel()/block);
  TORCH_CHECK(rank>=0&&rank<4&&peers.size()==4&&topk>0&&tokens*topk==rows&&tokens<=capacity,"tail shape/rank mismatch");
  TORCH_CHECK(activation.scalar_type()==torch::kBFloat16&&pair.scalar_type()==torch::kBFloat16&&output.scalar_type()==torch::kBFloat16&&shared.scalar_type()==torch::kBFloat16,"BF16 native tail required");
  TORCH_CHECK(weight.scalar_type()==torch::kInt32&&weight.dim()==3&&weight.size(0)==64&&weight.size(1)==k/16&&weight.size(2)==n*2,"native repacked MXFP4 layout differs");
  TORCH_CHECK(k%32==0&&k%(16*NT_K)==0&&n%(16*NT_N)==0,"native full-K geometry unsupported");
  TORCH_CHECK(pair.dim()==2&&pair.size(0)==rows&&pair.size(1)==n&&shared.sizes()==output.sizes(),"tail pair/shared output shape differs");
  TORCH_CHECK(routing.numel()==rows&&padded.numel()==1&&scales.nbytes()==size_t(64)*k*n/32,"native scale/routing geometry differs");
  TORCH_CHECK(state.scalar_type()==torch::kUInt8&&groups.scalar_type()==torch::kInt32&&stats.scalar_type()==torch::kInt64&&visits.scalar_type()==torch::kInt32&&trace.scalar_type()==torch::kInt64&&events.scalar_type()==torch::kInt64&&mode.scalar_type()==torch::kInt32&&anchors.scalar_type()==torch::kInt32,"tail state dtype mismatch");
  TORCH_CHECK(sorted.scalar_type()==torch::kInt32&&experts.scalar_type()==torch::kInt32&&padded.scalar_type()==torch::kInt32&&routing.scalar_type()==torch::kFloat32,"grouping/routing dtypes differ");
  TORCH_CHECK(state.nbytes()>=sizeof(ExecutorState)&&groups.numel()>=capacity_groups+(rows+31)/32&&stats.numel()>=64&&mode.numel()>=4,"tail state capacity differs");
  TORCH_CHECK(plans.dim()==2&&plans.size(1)==4&&plans.size(0)>0&&plans.size(0)<=kCostMaxPlans&&plans.scalar_type()==torch::kInt32,"tail plans invalid");
  TORCH_CHECK(table.scalar_type()==torch::kFloat64&&table.numel()==4*plans.size(0)*kCostAnchors*kCostRow&&anchors.numel()==4*plans.size(0),"tail measured table geometry invalid");
  TORCH_CHECK(!visits.numel()||visits.numel()>=capacity_groups*(n/(16*NT_N)),"tail task bitmap undersized");
  TORCH_CHECK(!trace.numel()||trace.numel()>=workers*10,"tail worker trace undersized");
  TORCH_CHECK(!events.numel()||events.numel()>=(capacity_groups*(n/(16*NT_N))+20)*4,"task interval storage undersized");
  const int device=activation.get_device();
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device));
  int clock_khz=0;check(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device));
  int active;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,resident_executor,NT_THREADS,scratch));
  TORCH_CHECK(workers>1&&workers<=active*prop.multiProcessorCount,"final executor resident capacity exceeded");
  ExecutorArgs a{};a.down={reinterpret_cast<int4*>(activation.data_ptr()),reinterpret_cast<int4*>(weight.data_ptr()),reinterpret_cast<int4*>(scales.data_ptr()),reinterpret_cast<int4*>(pair.data_ptr()),sorted.data_ptr<int>(),experts.data_ptr<int>(),padded.data_ptr<int>(),routing.data_ptr<float>(),rows,n,k,int(weight.size(0)),capacity_groups};
  a.transport.layout=Layout(capacity,n);for(int r=0;r<4;++r)a.transport.peers[r]=reinterpret_cast<void*>(peers[r]);
  a.transport.rank=rank;a.transport.tokens=tokens;a.transport.session=session;a.transport.layer=layer;
  a.transport.timeout_cycles=U64(timeout*clock_khz*1000.);a.transport.output=reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  a.shared=reinterpret_cast<__nv_bfloat16*>(shared.data_ptr());a.routed_scale=scale;a.topk=topk;a.block_rows=block;a.tiles=n/(16*NT_N);a.workers=workers;a.scratch_bytes=scratch;a.plan_count=plans.size(0);
  a.groups=groups.data_ptr<int>();a.state=reinterpret_cast<ExecutorState*>(state.data_ptr());a.stats=reinterpret_cast<U64*>(stats.data_ptr());
  a.visits=visits.numel()?visits.data_ptr<int>():nullptr;a.trace=trace.numel()?reinterpret_cast<U64*>(trace.data_ptr()):nullptr;
  a.events=events.numel()?reinterpret_cast<U64*>(events.data_ptr()):nullptr;
  a.mode=mode.data_ptr<int>();a.plans=reinterpret_cast<TailPlan*>(plans.data_ptr());a.table=table.data_ptr<double>();a.anchors=anchors.data_ptr<int>();
  void*args[]={&a};check(cudaLaunchCooperativeKernel((void*)resident_executor,workers,NT_THREADS,args,scratch,at::cuda::getCurrentCUDAStream()));
}
pybind11::dict snapshot(torch::Tensor tensor){
  c10::cuda::CUDAGuard guard(tensor.device());
  ExecutorState state;check(cudaMemcpy(&state,tensor.data_ptr(),sizeof(state),cudaMemcpyDeviceToHost));
  pybind11::dict d;d["local_groups"]=state.groups;d["features"]=std::vector<double>(state.features,state.features+5);d["selected_index"]=state.selected;
  d["chosen_indices"]=std::vector<int>(state.chosen,state.chosen+4);d["leader_prediction_us"]=state.prediction;
  d["start_ns"]=state.start;d["planned_ns"]=state.planned;d["finish_ns"]=state.finish;d["planner_begin_ns"]=state.planner_begin;d["planner_end_ns"]=state.planner_end;d["completion_begin_ns"]=state.completion_begin;d["role_switch_ns"]=state.role_switch;d["first_ready_ns"]=state.first_ready;
  std::vector<std::vector<U64>>times;for(int c=0;c<4;++c)times.emplace_back(state.times[c],state.times[c]+8);d["chunk_times_ns"]=times;return d;
}
void mark_front(torch::Tensor stats,torch::Tensor state,torch::Tensor mode){mark_front_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<U64*>(stats.data_ptr()),reinterpret_cast<ExecutorState*>(state.data_ptr()),mode.data_ptr<int>());check(cudaGetLastError());}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("resources",&resources);m.def("run",&launch);m.def("snapshot",&snapshot);m.def("mark_front",&mark_front);}
