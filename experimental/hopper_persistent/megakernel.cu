// Hopper M64 persistent IPC dispatch, WGMMA expert tiles, and origin combine.
// This backend has no staged operator, child kernel, or NCCL routed path.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "scheduler.h"
#include "tile_scheduler.h"
#include "hopper_tile.cuh"
#include "../../csrc/expert_numerics.h"

#define CUDA_OK(call) do { cudaError_t result=(call); TORCH_CHECK(result==cudaSuccess, "CUDA failure: ", cudaGetErrorString(result)); } while(0)
namespace {
using U64 = unsigned long long;
constexpr int MAX_EP=4, MAX_K=8, THREADS=128, EVIDENCE_FIELDS=16, EVIDENCE_PLANES=86;
struct RoundBF16 {
  __device__ float operator()(float x) const { return __bfloat162float(__float2bfloat16_rn(x)); }
};
struct alignas(256) Control {
  U64 ready,done,planned;
  int n,dispatch_next,dispatch_done,return_next,return_done,final_next,ctas_done;
  int npayloads,ntasks,local_pairs,remote_pairs,remote_unique;
  int c,k,steal;
  int actual_stolen,actual_computed;
  int actual_tiles,actual_stolen_tiles,ngroups;
  int active_compute,active_dispatch,return_compute_overlap,dispatch_compute_overlap;
  int workload[7],groups_by_chunks[MAX_K];
  int starts[MAX_K],ends[MAX_K],computed[MAX_K];
};
struct Payload { int origin,token; };
struct Task { int origin,token,slot,expert,payload,chunk; };
struct Group { int expert,chunk,rows,tasks[weave::TILE_ROWS]; };
struct View {
  Control* ctl;
  __nv_bfloat16* x;
  int* ids;
  float* routing;
  Payload* payloads;
  Task* tasks;
  __nv_bfloat16* cache;
  U64* cache_ready;
  float* task_out;
  float* returned;
  U64* returned_ready;
  __nv_bfloat16* output;
  Group* groups;
  weave::TileState* tile_states;
  int* group_heads;
  __nv_bfloat16* group_mid;
};
struct Config {
  View peers[MAX_EP];
  int rank,ep,capacity,h,i,topk,experts,workers;
  const unsigned char *s1,*s3,*s2;
  // __grid_constant__ preserves these descriptors in kernel parameter space.
  // Device helpers must receive Config by const reference, never by value.
  CUtensorMap gate_map,up_map,down_map;
  const double* profile;
  int nprofile,fixed_c,fixed_k,fixed_steal,calibration,max_groups;
  float clamp;
  U64 timeout;
  int* worker_smids;
  U64 *mechanism_last,*mechanism_evidence;
  int evidence_plane;
};
static_assert(offsetof(Config,gate_map)%64==0 && offsetof(Config,up_map)%64==0 && offsetof(Config,down_map)%64==0,
              "embedded TMA descriptors must retain 64-byte alignment");

struct Layout {
  size_t bytes=0,control,x,ids,routing,payloads,tasks,cache,cache_ready,task_out,returned,returned_ready,output;
  size_t groups,tile_states,group_heads,group_mid,worker_smids,mechanism_last,mechanism_evidence;
  size_t take(size_t n) { size_t p=(bytes+255)&~size_t(255); bytes=p+n; return p; }
  Layout(int ep,int cap,int h,int i,int topk,int local_experts,int workers=0) {
    size_t tokens=size_t(ep)*cap,pairs=tokens*topk;
    control=take(sizeof(Control));x=take(size_t(cap)*h*2);
    ids=take(size_t(cap)*topk*4);routing=take(size_t(cap)*topk*4);
    payloads=take(tokens*sizeof(Payload));tasks=take(pairs*sizeof(Task));
    cache=take(tokens*h*2);cache_ready=take(tokens*sizeof(U64));
    task_out=take(pairs*h*4);returned=take(size_t(cap)*topk*h*4);
    returned_ready=take(size_t(cap)*topk*sizeof(U64));output=take(size_t(cap)*h*2);
    size_t max_groups=weave::tile_group_capacity(int(pairs),local_experts,MAX_K);
    groups=take(max_groups*sizeof(Group));tile_states=take(max_groups*sizeof(weave::TileState));
    group_heads=take(size_t(local_experts)*MAX_K*sizeof(int));
    group_mid=take(max_groups*weave::TILE_ROWS*i*sizeof(__nv_bfloat16));
    worker_smids=take(size_t(workers)*sizeof(int));
    mechanism_last=take(EVIDENCE_FIELDS*sizeof(U64));
    mechanism_evidence=take(size_t(EVIDENCE_PLANES)*(EVIDENCE_FIELDS+workers*MAX_K)*sizeof(U64));
  }
  View view(void* ptr) const {
    auto p=static_cast<char*>(ptr);
    return {reinterpret_cast<Control*>(p+control),reinterpret_cast<__nv_bfloat16*>(p+x),
      reinterpret_cast<int*>(p+ids),reinterpret_cast<float*>(p+routing),
      reinterpret_cast<Payload*>(p+payloads),reinterpret_cast<Task*>(p+tasks),
      reinterpret_cast<__nv_bfloat16*>(p+cache),reinterpret_cast<U64*>(p+cache_ready),
      reinterpret_cast<float*>(p+task_out),reinterpret_cast<float*>(p+returned),
      reinterpret_cast<U64*>(p+returned_ready),reinterpret_cast<__nv_bfloat16*>(p+output),
      reinterpret_cast<Group*>(p+groups),reinterpret_cast<weave::TileState*>(p+tile_states),
      reinterpret_cast<int*>(p+group_heads),reinterpret_cast<__nv_bfloat16*>(p+group_mid)};
  }
};
__device__ U64 acquire(U64* p) {
  return cuda::atomic_ref<U64,cuda::thread_scope_system>(*p).load(cuda::memory_order_acquire);
}
__device__ void release(U64* p,U64 v) {
  cuda::atomic_ref<U64,cuda::thread_scope_system>(*p).store(v,cuda::memory_order_release);
}
__device__ int load_i(int* p) {
  return cuda::atomic_ref<int,cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}
__device__ int add_i(int* p,int x) {
  return cuda::atomic_ref<int,cuda::thread_scope_device>(*p).fetch_add(x,cuda::memory_order_acq_rel);
}
__device__ void check(bool good) { if(!good) asm volatile("trap;"); }
__device__ void wait_epoch(U64* p,U64 epoch,U64 timeout) {
  U64 start=clock64();
  while(acquire(p)<epoch) { check(clock64()-start<timeout);__nanosleep(64); }
}
__device__ void wait_count(int* p,int n,U64 timeout) {
  U64 start=clock64();
  while(load_i(p)<n) { check(clock64()-start<timeout);__nanosleep(64); }
}
// Every producing lane fences before the elected lane publishes completion.
__device__ void publish_cta_device() { __threadfence();__syncthreads(); }
__device__ void publish_cta_system() { __threadfence_system();__syncthreads(); }
__global__ void prepare(Control* p,int capacity,U64* last) {
  check(p->n>=0 && p->n<=capacity);
  p->dispatch_next=p->dispatch_done=p->return_next=p->return_done=p->final_next=p->ctas_done=0;
  p->npayloads=p->ntasks=p->local_pairs=p->remote_pairs=p->remote_unique=0;
  p->actual_stolen=p->actual_computed=p->actual_tiles=p->actual_stolen_tiles=p->ngroups=0;
  p->active_compute=p->active_dispatch=p->return_compute_overlap=p->dispatch_compute_overlap=0;
  for(int j=0;j<7;++j) p->workload[j]=0;
  for(int k=0;k<MAX_K;++k) p->starts[k]=p->ends[k]=p->computed[k]=p->groups_by_chunks[k]=0;
  for(int j=0;j<EVIDENCE_FIELDS;++j) last[j]=0;
  __threadfence_system();
  release(&p->ready,acquire(&p->ready)+1);
}
struct DeviceLoad { __device__ int operator()(int* p) const { return load_i(p); } };
struct DeviceAdd { __device__ int operator()(int* p,int v) const { return add_i(p,v); } };
struct GroupReady {
  View view; U64 epoch;
  __device__ bool operator()(int id) const {
    const Group& group=view.groups[id];
    for(int row=0;row<group.rows;++row)
      if(acquire(&view.cache_ready[view.tasks[group.tasks[row]].payload])<epoch) return false;
    return true;
  }
};
struct PersistentTileInput {
  View view; const Group* group; int group_id,h,i; bool down;
  __device__ __nv_bfloat16 operator()(int row,int column) const {
    if(down) return view.group_mid[(size_t(group_id)*weave::TILE_ROWS+row)*i+column];
    return view.cache[size_t(view.tasks[group->tasks[row]].payload)*h+column];
  }
};
struct PersistentMidStore {
  __nv_bfloat16* mid; int group_id,i;
  __device__ void operator()(int row,int column,__nv_bfloat16 value) const {
    mid[(size_t(group_id)*weave::TILE_ROWS+row)*i+column]=value;
  }
};
struct PersistentDownStore {
  float* output; const Group* group; int h;
  __device__ void operator()(int row,int column,float value) const {
    check(isfinite(value));output[size_t(group->tasks[row])*h+column]=value;
  }
};
__device__ void compute_tile(const Config& cfg,weave::TileClaim claim,weave::paper::Shared& scratch) {
  View view=cfg.peers[cfg.rank]; const Group& group=view.groups[claim.group];
  if(threadIdx.x==0) {
    add_i(&view.ctl->active_compute,1);
    if(load_i(&view.ctl->active_dispatch)>0) add_i(&view.ctl->dispatch_compute_overlap,1);
  }
  __syncthreads();
  PersistentTileInput input{view,&group,claim.group,cfg.h,cfg.i,claim.stage!=0};
  int first=claim.column*weave::TILE_COLUMNS;
  if(claim.stage==0)
    weave::paper::gate_up(&cfg.gate_map,&cfg.up_map,cfg.s1,cfg.s3,group.expert,group.rows,cfg.h,cfg.i,first,cfg.clamp,
      input,PersistentMidStore{view.group_mid,claim.group,cfg.i},scratch,cfg.calibration!=1);
  else
    weave::paper::down(&cfg.down_map,cfg.s2,group.expert,group.rows,cfg.h,cfg.i,first,
      input,PersistentDownStore{view.task_out,&group,cfg.h},scratch,cfg.calibration!=1);
  publish_cta_device();
  if(threadIdx.x==0) {
    add_i(&view.ctl->actual_tiles,1);
    auto state=&view.tile_states[claim.group];
    if(claim.stage==0) add_i(&state->gate_done,1);
    else if(add_i(&state->down_done,1)+1==weave::tile_columns(cfg.h)) {
      add_i(&view.ctl->computed[group.chunk],group.rows);
      add_i(&view.ctl->actual_computed,group.rows);
    }
    add_i(&view.ctl->active_compute,-1);
  }
  __syncthreads();
}
// Only the final completed worker publishes this whole-call record.
__device__ void record_mechanisms(const Config& cfg,Control* p) {
  check(p->actual_tiles==p->workload[6] && p->actual_computed==p->ntasks);
  check(p->dispatch_done==p->npayloads && p->return_done==p->ntasks);
  check(p->active_compute==0 && p->active_dispatch==0);
  for(int worker=0;worker<cfg.workers;++worker) {
    check(cfg.worker_smids[worker]>=0);
    for(int previous=0;previous<worker;++previous)
      check(cfg.worker_smids[previous]!=cfg.worker_smids[worker]);
  }
  U64 row[EVIDENCE_FIELDS]={1,U64(cfg.fixed_c<0),U64(p->actual_tiles),U64(p->actual_stolen_tiles),
    U64(p->dispatch_done),U64(p->remote_pairs),U64(p->remote_unique),U64(p->return_compute_overlap),
    U64(p->dispatch_compute_overlap),U64(p->c),U64(p->k),U64(p->steal),U64(p->n),
    U64(p->ntasks),U64(p->ngroups),U64(cfg.fixed_c<0?cfg.nprofile:0)};
  for(int j=0;j<EVIDENCE_FIELDS;++j) cfg.mechanism_last[j]=row[j];
  if(cfg.evidence_plane>=0) {
    U64* plane=cfg.mechanism_evidence+size_t(cfg.evidence_plane)*(EVIDENCE_FIELDS+cfg.workers*MAX_K);
    for(int j=0;j<EVIDENCE_FIELDS;++j) plane[j]+=row[j];
    ++plane[EVIDENCE_FIELDS+p->c*MAX_K+p->k-1];
  }
}
__global__ void persistent(const __grid_constant__ Config cfg) {
  View v=cfg.peers[cfg.rank];Control* p=v.ctl;
  U64 epoch=acquire(&p->ready);
  extern __shared__ __align__(1024) unsigned char shared_bytes[];
  auto& scratch=*reinterpret_cast<weave::paper::Shared*>(shared_bytes);
  __shared__ int job;
  __shared__ weave::TileClaim tile_job;
  if(threadIdx.x==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    cfg.worker_smids[blockIdx.x]=int(smid);
  }
  if(blockIdx.x==0 && threadIdx.x==0) {
    for(int origin=0;origin<cfg.ep;++origin) {
      View from=cfg.peers[origin];wait_epoch(&from.ctl->ready,epoch,cfg.timeout);
      int n=from.ctl->n;check(n>=0 && n<=cfg.capacity);
      for(int token=0;token<n;++token) {
        int payload=-1;
        for(int slot=0;slot<cfg.topk;++slot) {
          int expert=from.ids[token*cfg.topk+slot];float w=from.routing[token*cfg.topk+slot];
          check(expert>=0 && expert<cfg.experts && isfinite(w) && w>=0);
          for(int prev=0;prev<slot;++prev) check(from.ids[token*cfg.topk+prev]!=expert);
          if(expert/(cfg.experts/cfg.ep)!=cfg.rank) continue;
          if(payload<0) { payload=p->npayloads++;v.payloads[payload]={origin,token};if(origin!=cfg.rank) ++p->remote_unique; }
          v.tasks[p->ntasks++]={origin,token,slot,expert%(cfg.experts/cfg.ep),payload,0};
          if(origin==cfg.rank) ++p->local_pairs;else ++p->remote_pairs;
        }
      }
    }
    if(cfg.fixed_c<0 && p->ntasks>0) {
      for(int chunks=1;chunks<=MAX_K && chunks<=p->npayloads;++chunks) {
        bool requested=false;
        for(int j=0;j<cfg.nprofile;++j) if(cfg.profile[j*6+1]==chunks) requested=true;
        if(!requested) continue;
        for(int expert=0;expert<cfg.experts/cfg.ep;++expert)
          for(int k=0;k<chunks;++k) v.group_heads[expert*MAX_K+k]=0;
        for(int task=0;task<p->ntasks;++task) {
          Task t=v.tasks[task];++v.group_heads[t.expert*MAX_K+t.payload*chunks/p->npayloads];
        }
        int groups=0;
        for(int expert=0;expert<cfg.experts/cfg.ep;++expert)
          for(int k=0;k<chunks;++k) groups+=(v.group_heads[expert*MAX_K+k]+weave::TILE_ROWS-1)/weave::TILE_ROWS;
        p->groups_by_chunks[chunks-1]=groups;
      }
    }
    weave::Schedule s;
    if(cfg.fixed_c>=0) {
      s={cfg.fixed_c,cfg.fixed_k,cfg.fixed_steal,0.};
      if(p->npayloads<s.k) s.k=p->npayloads>0?p->npayloads:1;
      check(s.c>0 || p->remote_pairs==0);
    } else s=weave::choose_schedule(cfg.profile,cfg.nprofile,cfg.workers,p->local_pairs,
      p->remote_pairs,p->remote_unique,p->npayloads,cfg.h,cfg.i,p->groups_by_chunks);
    check(s.c>=0 && s.c<cfg.workers && s.k>=1 && s.k<=MAX_K);
    p->c=s.c;p->k=s.k;p->steal=s.steal;
    for(int task=0;task<p->ntasks;++task) {
      int k=v.tasks[task].payload*s.k/p->npayloads;
      v.tasks[task].chunk=k;++p->ends[k];
      // Payload enumeration guarantees contiguous ascending chunk tasks.
      check(task==0 || v.tasks[task-1].chunk<=k);
    }
    int start=0;
    for(int k=0;k<s.k;++k) { int count=p->ends[k];p->starts[k]=start;start+=count;p->ends[k]=start; }
    check(cfg.calibration!=2 || p->remote_pairs==0);
    for(int expert=0;expert<cfg.experts/cfg.ep;++expert)
      for(int k=0;k<s.k;++k) v.group_heads[expert*MAX_K+k]=-1;
    for(int task=0;task<p->ntasks;++task) {
      Task t=v.tasks[task];int bucket=t.expert*MAX_K+t.chunk;
      int group=v.group_heads[bucket];
      if(group<0 || v.groups[group].rows==weave::TILE_ROWS) {
        group=p->ngroups++;check(group<cfg.max_groups);
        v.group_heads[bucket]=group;v.groups[group].expert=t.expert;v.groups[group].chunk=t.chunk;
        v.groups[group].rows=0;v.tile_states[group]={0,0,0,0};
        check(group==0 || v.groups[group-1].chunk<=t.chunk);
      }
      v.groups[group].tasks[v.groups[group].rows++]=task;
    }
    if(cfg.fixed_c<0 && p->ngroups>0) check(p->ngroups==p->groups_by_chunks[p->k-1]);
    p->workload[0]=p->local_pairs;p->workload[1]=p->remote_pairs;p->workload[2]=p->remote_unique;
    p->workload[3]=p->npayloads;p->workload[4]=p->ntasks;p->workload[5]=p->ngroups;
    p->workload[6]=p->ngroups*(weave::tile_columns(cfg.i)+weave::tile_columns(cfg.h));
    release(&p->planned,epoch);
  }
  if(threadIdx.x==0) wait_epoch(&p->planned,epoch,cfg.timeout);
  __syncthreads();
  bool comm=blockIdx.x<unsigned(p->c);
  // Pull every unique inbound token once. Compute workers can consume a ready
  // group while communication workers continue dispatching later payloads.
  if(comm || p->c==0) {
    while(true) {
      if(threadIdx.x==0) {
        job=add_i(&p->dispatch_next,1);
        if(job<p->npayloads) {
          add_i(&p->active_dispatch,1);
          if(load_i(&p->active_compute)>0) add_i(&p->dispatch_compute_overlap,1);
        }
      }
      __syncthreads();if(job>=p->npayloads) break;
      Payload token=v.payloads[job];View from=cfg.peers[token.origin];
      for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x) {
        auto value=from.x[size_t(token.token)*cfg.h+col];
        check(isfinite(__bfloat162float(value)));v.cache[size_t(job)*cfg.h+col]=value;
      }
      publish_cta_device();
      if(threadIdx.x==0) {
        release(&v.cache_ready[job],epoch);add_i(&p->dispatch_done,1);add_i(&p->active_dispatch,-1);
      }
      __syncthreads();
    }
    if(threadIdx.x==0) wait_count(&p->dispatch_done,p->npayloads,cfg.timeout);
    __syncthreads();
  }
  int budget=comm?p->steal:0x7fffffff;
  U64 progress=clock64();
  for(int completed=0;completed<budget;) {
    if(threadIdx.x==0) tile_job=weave::claim_tile(v.tile_states,p->ngroups,
      weave::tile_columns(cfg.i),weave::tile_columns(cfg.h),DeviceLoad{},DeviceAdd{},GroupReady{v,epoch});
    __syncthreads();
    if(tile_job.group==-1) break;
    if(tile_job.group==-2) {
      // A stealing worker never waits for a compute dependency; it starts
      // chunk returns. Dedicated compute workers remain until all tiles finish.
      if(comm) break;
      check(clock64()-progress<cfg.timeout);__nanosleep(64);__syncthreads();continue;
    }
    compute_tile(cfg,tile_job,scratch);
    if(comm && threadIdx.x==0) { add_i(&p->actual_stolen,1);add_i(&p->actual_stolen_tiles,1); }
    ++completed;progress=clock64();
  }
  // Communication workers wait only for their owning chunk. Compute workers
  // join this queue after the compute graph drains, preserving forward progress.
  while(true) {
    if(threadIdx.x==0) job=add_i(&p->return_next,1);
    __syncthreads();if(job>=p->ntasks) break;
    Task task=v.tasks[job];
    if(threadIdx.x==0) wait_count(&p->computed[task.chunk],p->ends[task.chunk]-p->starts[task.chunk],cfg.timeout);
    __syncthreads();
    if(threadIdx.x==0 && load_i(&p->active_compute)>0) add_i(&p->return_compute_overlap,1);
    View to=cfg.peers[task.origin];size_t slot=size_t(task.token)*cfg.topk+task.slot;
    for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x)
      to.returned[slot*cfg.h+col]=v.task_out[size_t(job)*cfg.h+col];
    publish_cta_system();
    if(threadIdx.x==0) { release(&to.returned_ready[slot],epoch);add_i(&p->return_done,1); }
    __syncthreads();
  }
  // No resident CTA waits for incoming results while this rank owes returns.
  if(threadIdx.x==0) wait_count(&p->return_done,p->ntasks,cfg.timeout);
  __syncthreads();
  while(true) {
    if(threadIdx.x==0) job=add_i(&p->final_next,1);
    __syncthreads();if(job>=p->n) break;
    if(threadIdx.x==0) for(int slot=0;slot<cfg.topk;++slot)
      wait_epoch(&v.returned_ready[size_t(job)*cfg.topk+slot],epoch,cfg.timeout);
    __syncthreads();
    for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x) {
      float sum=0.f;
      for(int slot=0;slot<cfg.topk;++slot) {
        size_t pos=size_t(job)*cfg.topk+slot;
        sum+=weave::routed_product(v.returned[pos*cfg.h+col],v.routing[pos],RoundBF16{});
      }
      auto result=__float2bfloat16_rn(sum);check(isfinite(__bfloat162float(result)));
      v.output[size_t(job)*cfg.h+col]=result;
    }
    __syncthreads();
  }
  publish_cta_system();
  if(threadIdx.x==0) {
    if(add_i(&p->ctas_done,1)==cfg.workers-1) {
      record_mechanisms(cfg,p);release(&p->done,epoch);
    }
    for(int peer=0;peer<cfg.ep;++peer) wait_epoch(&cfg.peers[peer].ctl->done,epoch,cfg.timeout);
  }
}

class Workspace;
struct BoundWeights {
  std::shared_ptr<Workspace> owner;
  std::vector<torch::Tensor> weights;
  Config config;
  BoundWeights(std::shared_ptr<Workspace> owner_,const std::vector<torch::Tensor>& weights_,Config config_)
    :owner(std::move(owner_)),weights(weights_),config(config_) {}
};
class Workspace : public std::enable_shared_from_this<Workspace> {
 public:
  int rank,ep,cap,h,i,topk,experts,device,workers;
  int clock_khz=0,dynamic_shared_bytes=0,active_blocks_per_sm=0;
  size_t shared_bytes_per_sm=0;
  Layout layout;
  void* base=nullptr;
  void* pointers[MAX_EP]{};
  bool connected=false;
  cudaFuncAttributes kernel_attributes{};
  Config static_config{};
  // The convenience unbound run API also retains checkpoint storage.
  std::vector<torch::Tensor> unbound_weights;
  Workspace(int rank_,int ep_,int cap_,int h_,int i_,int topk_,int experts_)
    :rank(rank_),ep(ep_),cap(cap_),h(h_),i(i_),topk(topk_),experts(experts_),layout(1,1,128,128,1,1) {
    TORCH_CHECK(ep==MAX_EP && rank>=0 && rank<ep && cap>0 && cap<=8192,"Hopper persistent requires EP4 and capacity in [1,8192]");
    TORCH_CHECK(h>0 && h%128==0 && h<=8192 && i>0 && i%128==0 && i<=8192,"H/I must be positive multiples of 128, at most 8192");
    TORCH_CHECK(topk>0 && topk<=16 && experts>=topk && experts%ep==0,"invalid expert/top-k shape");
    CUDA_OK(cudaGetDevice(&device));
    TORCH_CHECK(device==rank,"persistent IPC requires CUDA device ordinal == EP rank");
    cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.major==9 && prop.minor==0,"Hopper persistent requires SM90a");
    workers=prop.multiProcessorCount;TORCH_CHECK(workers>1,"at least two SMs required");
    shared_bytes_per_sm=prop.sharedMemPerMultiprocessor;
    dynamic_shared_bytes=int((shared_bytes_per_sm/2+1024)&~size_t(1023));
    if(dynamic_shared_bytes<int(sizeof(weave::paper::Shared))) dynamic_shared_bytes=sizeof(weave::paper::Shared);
    TORCH_CHECK(size_t(dynamic_shared_bytes)>shared_bytes_per_sm/2,"shared reservation must enforce one CTA per SM");
    TORCH_CHECK(dynamic_shared_bytes<=int(prop.sharedMemPerBlockOptin),"device shared memory cannot fit resident Hopper tile");
    CUDA_OK(cudaFuncSetAttribute(persistent,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_shared_bytes));
    CUDA_OK(cudaFuncGetAttributes(&kernel_attributes,persistent));
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,persistent,THREADS,dynamic_shared_bytes));
    TORCH_CHECK(active_blocks_per_sm==1,"persistent protocol requires exactly one resident CTA per SM");
    CUDA_OK(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device));
    TORCH_CHECK(clock_khz>0,"device clock rate unavailable");
    layout=Layout(ep,cap,h,i,topk,experts/ep,workers);
    CUDA_OK(cudaMalloc(&base,layout.bytes));
    auto stream=at::cuda::getCurrentCUDAStream(device).stream();
    CUDA_OK(cudaMemsetAsync(base,0,layout.bytes,stream));
    CUDA_OK(cudaMemsetAsync(static_cast<char*>(base)+layout.worker_smids,0xff,size_t(workers)*sizeof(int),stream));
    CUDA_OK(cudaStreamSynchronize(stream));pointers[rank]=base;
  }
  ~Workspace() {
    if(!base) return;
    c10::cuda::CUDAGuard guard(device);
    for(int p=0;p<ep;++p) if(p!=rank && pointers[p]) cudaIpcCloseMemHandle(pointers[p]);
    cudaFree(base);
  }
  pybind11::bytes handle() {
    c10::cuda::CUDAGuard guard(device);cudaIpcMemHandle_t handle;
    CUDA_OK(cudaIpcGetMemHandle(&handle,base));
    return pybind11::bytes(reinterpret_cast<char*>(&handle),sizeof(handle));
  }
  void connect(const std::vector<std::string>& handles) {
    TORCH_CHECK(!connected && int(handles.size())==ep,"invalid or duplicate IPC peer connection");
    c10::cuda::CUDAGuard guard(device);
    for(int p=0;p<ep;++p) {
      TORCH_CHECK(handles[p].size()==sizeof(cudaIpcMemHandle_t),"invalid IPC handle size");
      if(p==rank) continue;
      int access=0;CUDA_OK(cudaDeviceCanAccessPeer(&access,device,p));TORCH_CHECK(access,"peer access unavailable");
      int atomics=0;CUDA_OK(cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,device,p));
      TORCH_CHECK(atomics,"system-scope native peer atomics required");
      cudaIpcMemHandle_t handle;memcpy(&handle,handles[p].data(),sizeof(handle));
      CUDA_OK(cudaIpcOpenMemHandle(&pointers[p],handle,cudaIpcMemLazyEnablePeerAccess));
    }
    for(int p=0;p<ep;++p) static_config.peers[p]=layout.view(pointers[p]);
    static_config.rank=rank;static_config.ep=ep;static_config.capacity=cap;
    static_config.h=h;static_config.i=i;static_config.topk=topk;
    static_config.experts=experts;static_config.workers=workers;
    static_config.max_groups=weave::tile_group_capacity(ep*cap*topk,experts/ep,MAX_K);
    static_config.worker_smids=reinterpret_cast<int*>(static_cast<char*>(base)+layout.worker_smids);
    static_config.mechanism_last=reinterpret_cast<U64*>(static_cast<char*>(base)+layout.mechanism_last);
    static_config.mechanism_evidence=reinterpret_cast<U64*>(static_cast<char*>(base)+layout.mechanism_evidence);
    static_config.evidence_plane=-1;
    connected=true;
  }
  std::shared_ptr<BoundWeights> bind_weights(const std::vector<torch::Tensor>& w) {
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(connected && w.size()==6,"bind six raw tensors after peer connection");
    cudaStreamCaptureStatus capturing;CUDA_OK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(device).stream(),&capturing));
    TORCH_CHECK(capturing==cudaStreamCaptureStatusNone,"bind Hopper tensor maps before graph capture");
    std::vector<std::vector<int64_t>> shapes={{experts/ep,i,h/2},{experts/ep,i,h/2},{experts/ep,h,i/2},
      {experts/ep,i,h/32},{experts/ep,i,h/32},{experts/ep,h,i/32}};
    for(int j=0;j<6;++j) TORCH_CHECK(w[j].is_cuda() && w[j].get_device()==device && w[j].is_contiguous() &&
      w[j].scalar_type()==torch::kUInt8 && w[j].sizes().vec()==shapes[j],"bad raw MXFP4 tensor ",j);
    Config cfg=static_config;
    cfg.s1=w[3].data_ptr<unsigned char>();cfg.s3=w[4].data_ptr<unsigned char>();cfg.s2=w[5].data_ptr<unsigned char>();
    cfg.gate_map=weave::paper::tensor_map(w[0].data_ptr(),h/2,i,experts/ep,64);
    cfg.up_map=weave::paper::tensor_map(w[1].data_ptr(),h/2,i,experts/ep,64);
    cfg.down_map=weave::paper::tensor_map(w[2].data_ptr(),i/2,h,experts/ep,64);
    return std::make_shared<BoundWeights>(shared_from_this(),w,cfg);
  }
  pybind11::dict resources() const {
    namespace py=pybind11;py::dict result,dimensions,attr;
    dimensions["rank"]=rank;dimensions["ep"]=ep;dimensions["capacity"]=cap;
    dimensions["hidden_size"]=h;dimensions["intermediate_size"]=i;dimensions["top_k"]=topk;dimensions["experts"]=experts;
    attr["numRegs"]=kernel_attributes.numRegs;attr["localSizeBytes"]=kernel_attributes.localSizeBytes;
    attr["sharedSizeBytes"]=kernel_attributes.sharedSizeBytes;attr["constSizeBytes"]=kernel_attributes.constSizeBytes;
    attr["maxThreadsPerBlock"]=kernel_attributes.maxThreadsPerBlock;attr["binaryVersion"]=kernel_attributes.binaryVersion;
    attr["ptxVersion"]=kernel_attributes.ptxVersion;attr["maxDynamicSharedSizeBytes"]=kernel_attributes.maxDynamicSharedSizeBytes;
    result["implementation"]="hopper_persistent_paper_v1";result["device"]=device;result["clock_khz"]=clock_khz;
    result["dimensions"]=dimensions;result["workspace_bytes"]=layout.bytes;result["workers"]=workers;result["sm_count"]=workers;
    result["one_cta_per_sm"]=true;result["active_blocks_per_sm"]=active_blocks_per_sm;result["grid_blocks"]=workers;
    result["threads_per_block"]=THREADS;result["dynamic_shared_bytes"]=dynamic_shared_bytes;
    result["shared_bytes_per_sm"]=shared_bytes_per_sm;result["tile_shared_bytes"]=sizeof(weave::paper::Shared);
    result["hopper_device_compute"]=true;result["tile_rows"]=weave::TILE_ROWS;result["tile_columns"]=weave::TILE_COLUMNS;
    result["a_quantization"]="per_output_tile_k128";result["kernel_attributes"]=attr;
    result["numerical_policy"]="record_only";return result;
  }
  torch::Tensor tensor(size_t offset,std::vector<int64_t> shape,torch::ScalarType dtype) {
    auto owner=shared_from_this();
    return torch::from_blob(static_cast<char*>(base)+offset,shape,[owner](void*){},
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA,device));
  }
  torch::Tensor x() { return tensor(layout.x,{cap,h},torch::kBFloat16); }
  torch::Tensor ids() { return tensor(layout.ids,{cap,topk},torch::kInt32); }
  torch::Tensor routing() { return tensor(layout.routing,{cap,topk},torch::kFloat32); }
  torch::Tensor n() { return tensor(layout.control+offsetof(Control,n),{1},torch::kInt32); }
  torch::Tensor output() { return tensor(layout.output,{cap,h},torch::kBFloat16); }
  torch::Tensor schedule() { return tensor(layout.control+offsetof(Control,c),{3},torch::kInt32); }
  torch::Tensor work_counters() { return tensor(layout.control+offsetof(Control,actual_stolen),{2},torch::kInt32); }
  torch::Tensor workload() { return tensor(layout.control+offsetof(Control,workload),{7},torch::kInt32); }
  torch::Tensor tile_counters() { return tensor(layout.control+offsetof(Control,actual_tiles),{2},torch::kInt32); }
  torch::Tensor worker_smids() { return tensor(layout.worker_smids,{workers},torch::kInt32); }
  torch::Tensor mechanism_last() { return tensor(layout.mechanism_last,{EVIDENCE_FIELDS},torch::kInt64); }
  torch::Tensor mechanism_evidence() { return tensor(layout.mechanism_evidence,{EVIDENCE_PLANES,EVIDENCE_FIELDS+workers*MAX_K},torch::kInt64); }
  void run(const std::vector<torch::Tensor>& w,torch::Tensor profile,int c,int k,int steal,double timeout_seconds,
           float clamp,int tensor_core,int calibration,int trace_layer,int trace_phase) {
    auto bound=bind_weights(w);unbound_weights=w;
    run_bound(bound,profile,c,k,steal,timeout_seconds,clamp,tensor_core,calibration,trace_layer,trace_phase);
  }
  void run_bound(const std::shared_ptr<BoundWeights>& bound,torch::Tensor profile,int c,int k,int steal,
                 double timeout_seconds,float clamp,int tensor_core,int calibration,int trace_layer,int trace_phase) {
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(bound && bound->owner.get()==this,"bound weights belong to a different workspace");
    TORCH_CHECK(connected && std::isfinite(timeout_seconds) && timeout_seconds>0 && timeout_seconds<=3600 &&
      std::isfinite(clamp) && clamp>0,"invalid timeout or activation clamp");
    TORCH_CHECK(c>=-1 && c<workers && k>=1 && k<=MAX_K && steal>=0,"invalid c/K/steal schedule");
    TORCH_CHECK(tensor_core==2,"Hopper persistent supports only tiled device WGMMA compute");
    TORCH_CHECK(calibration>=0 && calibration<=2 && (!calibration || c>=0),"calibration requires fixed-plan execution");
    TORCH_CHECK(profile.is_cuda() && profile.get_device()==device && profile.is_contiguous() &&
      profile.scalar_type()==torch::kFloat64 && profile.dim()==2 && profile.size(1)==6,"profile must be CUDA float64 [N,6]");
    TORCH_CHECK(profile.size(0)<=workers*MAX_K && (c>=0 || profile.size(0)>0),"dynamic scheduling requires measured profile rows");
    TORCH_CHECK(trace_layer>=-1 && trace_layer<43 && trace_phase<=1 &&
      (trace_layer<0?trace_phase>=-1:trace_phase>=0),"invalid mechanism evidence layer/phase");
    TORCH_CHECK(active_blocks_per_sm==1,"persistent residency contract changed");
    Config cfg=bound->config;
    cfg.profile=profile.data_ptr<double>();cfg.nprofile=int(profile.size(0));cfg.fixed_c=c;cfg.fixed_k=k;cfg.fixed_steal=steal;
    cfg.calibration=calibration;cfg.timeout=U64(timeout_seconds*clock_khz*1000.);cfg.clamp=clamp;
    cfg.evidence_plane=trace_layer<0?-1:trace_phase*43+trace_layer;
    auto stream=at::cuda::getCurrentCUDAStream(device).stream();
    prepare<<<1,1,0,stream>>>(cfg.peers[rank].ctl,cap,cfg.mechanism_last);
    persistent<<<workers,THREADS,dynamic_shared_bytes,stream>>>(cfg);
    CUDA_OK(cudaGetLastError());
  }
};
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<BoundWeights,std::shared_ptr<BoundWeights>>(m,"BoundWeights");
  pybind11::class_<Workspace,std::shared_ptr<Workspace>>(m,"Workspace")
    .def(pybind11::init<int,int,int,int,int,int,int>())
    .def("handle",&Workspace::handle).def("connect",&Workspace::connect)
    .def_property_readonly("x",&Workspace::x).def_property_readonly("ids",&Workspace::ids)
    .def_property_readonly("routing",&Workspace::routing).def_property_readonly("n",&Workspace::n)
    .def_property_readonly("output",&Workspace::output).def_property_readonly("schedule",&Workspace::schedule)
    .def_property_readonly("work_counters",&Workspace::work_counters)
    .def_property_readonly("workload",&Workspace::workload).def_property_readonly("tile_counters",&Workspace::tile_counters)
    .def_property_readonly("worker_smids",&Workspace::worker_smids)
    .def_property_readonly("mechanism_last",&Workspace::mechanism_last)
    .def_property_readonly("mechanism_evidence",&Workspace::mechanism_evidence)
    .def("bind_weights",&Workspace::bind_weights).def("resources",&Workspace::resources)
    .def("run",&Workspace::run).def("run_bound",&Workspace::run_bound);
}
