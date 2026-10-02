// Persistent EP kernel with scalar and experimental BF16 WMMA expert work.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "scheduler.h"
#include "expert_numerics.h"

#define CUDA_OK(call) do { cudaError_t result=(call); TORCH_CHECK(result==cudaSuccess, "CUDA failure: ", cudaGetErrorString(result)); } while(0)
namespace {
using U64 = unsigned long long;
constexpr int MAX_EP = 4, MAX_K = 8, THREADS = 128;
struct RoundBF16 {
  __device__ __forceinline__ float operator()(float value) const {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
};
struct alignas(256) Control {
  U64 ready, done, planned;
  int n, dispatch_next, dispatch_done, return_next, return_done, final_next, ctas_done;
  int npayloads, ntasks, local_pairs, remote_pairs, remote_unique;
  int c, k, steal;
  int actual_stolen, actual_computed;
  int starts[MAX_K], ends[MAX_K], next[MAX_K], computed[MAX_K];
};
struct Payload { int origin, token; };
struct Task { int origin, token, slot, expert, payload, chunk; };
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
};
struct Config {
  View peers[MAX_EP];
  int rank, ep, capacity, h, i, topk, experts, workers;
  const unsigned char *w1, *w3, *w2, *s1, *s3, *s2;
  const double* profile;
  int nprofile, fixed_c, fixed_k, fixed_steal, tensor_core;
  float clamp;
  U64 timeout;
};

struct Layout {
  size_t bytes = 0, control, x, ids, routing, payloads, tasks, cache, cache_ready, task_out, returned, returned_ready, output;
  size_t take(size_t n) { size_t p = (bytes+255)&~size_t(255); bytes=p+n; return p; }
  Layout(int ep, int cap, int h, int topk) {
    size_t tokens=size_t(ep)*cap, pairs=tokens*topk;
    control=take(sizeof(Control)); x=take(size_t(cap)*h*2);
    ids=take(size_t(cap)*topk*4); routing=take(size_t(cap)*topk*4);
    payloads=take(tokens*sizeof(Payload)); tasks=take(pairs*sizeof(Task));
    cache=take(tokens*h*2); cache_ready=take(tokens*sizeof(U64));
    task_out=take(pairs*h*4); returned=take(size_t(cap)*topk*h*4);
    returned_ready=take(size_t(cap)*topk*sizeof(U64)); output=take(size_t(cap)*h*2);
  }
  View view(void* ptr) const {
    auto p=static_cast<char*>(ptr);
    return {reinterpret_cast<Control*>(p+control), reinterpret_cast<__nv_bfloat16*>(p+x),
      reinterpret_cast<int*>(p+ids), reinterpret_cast<float*>(p+routing),
      reinterpret_cast<Payload*>(p+payloads), reinterpret_cast<Task*>(p+tasks),
      reinterpret_cast<__nv_bfloat16*>(p+cache), reinterpret_cast<U64*>(p+cache_ready),
      reinterpret_cast<float*>(p+task_out), reinterpret_cast<float*>(p+returned),
      reinterpret_cast<U64*>(p+returned_ready), reinterpret_cast<__nv_bfloat16*>(p+output)};
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
__device__ void check(bool good) { if (!good) asm volatile("trap;"); }
__device__ void wait_epoch(U64* p,U64 epoch,U64 timeout) {
  U64 start=clock64();
  while(acquire(p)<epoch) { check(clock64()-start<timeout); __nanosleep(64); }
}
__device__ void wait_count(int* p,int n,U64 timeout) {
  U64 start=clock64();
  while(load_i(p)<n) { check(clock64()-start<timeout); __nanosleep(64); }
}
__global__ void prepare(Control* p,int capacity) {
  check(p->n>=0 && p->n<=capacity);
  p->dispatch_next=p->dispatch_done=p->return_next=p->return_done=p->final_next=p->ctas_done=0;
  p->npayloads=p->ntasks=p->local_pairs=p->remote_pairs=p->remote_unique=0;
  p->actual_stolen=p->actual_computed=0;
  for(int k=0;k<MAX_K;++k) p->starts[k]=p->ends[k]=p->next[k]=p->computed[k]=0;
  __threadfence_system();
  release(&p->ready,acquire(&p->ready)+1);
}
__device__ float decode4(unsigned char byte,int nibble,unsigned char scale) {
  int code=(byte>>(4*nibble))&15;
  constexpr float values[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float v=values[code&7];
  if(code&8) v=-v;
  return scale==255 ? CUDART_NAN_F : ldexpf(v,int(scale)-127);
}
__device__ float weight(const unsigned char* w,const unsigned char* s,size_t row,int col,int width) {
  return decode4(w[row*(width/2)+col/2],col&1,s[row*(width/32)+col/32]);
}
__device__ float warp_sum(float x) {
  for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
  return x;
}
__device__ int claim_task(Control* p) {
  for(int k=0;k<p->k;++k) {
    int j=add_i(&p->next[k],1);
    if(j<p->ends[k]) return j;
  }
  return -1;
}
// Each warp computes sixteen output rows. Only A's first token row is valid;
// the other fifteen rows are padded. This is a numerical Tensor Core candidate,
// not yet an efficient grouped-token GEMM or tile-granularity stealing scheme.
__device__ float wmma_dot(const __nv_bfloat16* x,const unsigned char* w,
                         const unsigned char* s,size_t first_row,int width,
                         __nv_bfloat16* scratch) {
  using namespace nvcuda;
  int lane=threadIdx.x%32;
  auto a=scratch; auto b=a+256;
  auto out=reinterpret_cast<float*>(b+256);
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> accum;
  wmma::fill_fragment(accum,0.f);
  for(int col=0;col<width;col+=16) {
    for(int j=lane;j<256;j+=32) {
      a[j]=j<16?x[col+j]:__float2bfloat16_rn(0.f);
      b[j]=__float2bfloat16_rn(weight(w,s,first_row+j/16,col+j%16,width));
    }
    __syncwarp();
    wmma::load_matrix_sync(af,a,16); wmma::load_matrix_sync(bf,b,16);
    wmma::mma_sync(accum,af,bf,accum);
    __syncwarp();
  }
  wmma::store_matrix_sync(out,accum,16,wmma::mem_row_major);
  __syncwarp();
  float result=lane<16?out[lane]:0.f;
  __syncwarp();
  return result;
}
__device__ void compute_task(Config cfg,int idx,U64 epoch,__nv_bfloat16* mid) {
  View v=cfg.peers[cfg.rank]; Task t=v.tasks[idx];
  if(threadIdx.x==0) wait_epoch(&v.cache_ready[t.payload],epoch,cfg.timeout);
  __syncthreads();
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const __nv_bfloat16* x=v.cache+size_t(t.payload)*cfg.h;
  if(cfg.tensor_core) {
    // 1024 BF16 elements/warp: A (256), B (256), C (256 floats).
    auto scratch=mid+cfg.i+warp*1024;
    for(int row=warp*16;row<cfg.i;row+=(THREADS/32)*16) {
      size_t wr=size_t(t.expert)*cfg.i+row;
      float g=wmma_dot(x,cfg.w1,cfg.s1,wr,cfg.h,scratch);
      float u=wmma_dot(x,cfg.w3,cfg.s3,wr,cfg.h,scratch);
      if(lane<16) {
        g=__bfloat162float(__float2bfloat16_rn(g));
        u=__bfloat162float(__float2bfloat16_rn(u));
        mid[row+lane]=__float2bfloat16_rn(weave::swiglu_product(g,u,cfg.clamp,RoundBF16{}));
      }
    }
    __syncthreads();
    for(int row=warp*16;row<cfg.h;row+=(THREADS/32)*16) {
      float sum=wmma_dot(mid,cfg.w2,cfg.s2,size_t(t.expert)*cfg.h+row,cfg.i,scratch);
      if(lane<16) v.task_out[size_t(idx)*cfg.h+row+lane]=__bfloat162float(__float2bfloat16_rn(sum));
    }
  } else {
  for(int row=warp;row<cfg.i;row+=THREADS/32) {
    float g=0.f,u=0.f; size_t wr=size_t(t.expert)*cfg.i+row;
    for(int col=lane;col<cfg.h;col+=32) {
      float xv=__bfloat162float(x[col]);
      g+=xv*weight(cfg.w1,cfg.s1,wr,col,cfg.h);
      u+=xv*weight(cfg.w3,cfg.s3,wr,col,cfg.h);
    }
    g=warp_sum(g); u=warp_sum(u);
    if(lane==0) {
      g=__bfloat162float(__float2bfloat16_rn(g));
      u=__bfloat162float(__float2bfloat16_rn(u));
      mid[row]=__float2bfloat16_rn(weave::swiglu_product(g,u,cfg.clamp,RoundBF16{}));
    }
  }
  __syncthreads();
  for(int row=warp;row<cfg.h;row+=THREADS/32) {
    float sum=0.f; size_t wr=size_t(t.expert)*cfg.h+row;
    for(int col=lane;col<cfg.i;col+=32)
      sum+=__bfloat162float(mid[col])*weight(cfg.w2,cfg.s2,wr,col,cfg.i);
    sum=warp_sum(sum);
    if(lane==0) v.task_out[size_t(idx)*cfg.h+row]=__bfloat162float(__float2bfloat16_rn(sum));
  }
  }
  __syncthreads();
  __threadfence();
  if(threadIdx.x==0) { add_i(&v.ctl->computed[t.chunk],1); add_i(&v.ctl->actual_computed,1); }
  __syncthreads();
}

__global__ void persistent(Config cfg) {
  View v=cfg.peers[cfg.rank]; Control* p=v.ctl;
  U64 epoch=acquire(&p->ready);
  extern __shared__ __align__(32) unsigned char shared_bytes[];
  auto mid=reinterpret_cast<__nv_bfloat16*>(shared_bytes);
  __shared__ int job;
  if(blockIdx.x==0 && threadIdx.x==0) {
    for(int origin=0;origin<cfg.ep;++origin) {
      View from=cfg.peers[origin]; wait_epoch(&from.ctl->ready,epoch,cfg.timeout);
      int n=from.ctl->n; check(n>=0 && n<=cfg.capacity);
      for(int token=0;token<n;++token) {
        int payload=-1;
        for(int slot=0;slot<cfg.topk;++slot) {
          int expert=from.ids[token*cfg.topk+slot];
          float w=from.routing[token*cfg.topk+slot];
          check(expert>=0 && expert<cfg.experts && isfinite(w) && w>=0);
          for(int prev=0;prev<slot;++prev) check(from.ids[token*cfg.topk+prev]!=expert);
          if(expert/(cfg.experts/cfg.ep)!=cfg.rank) continue;
          if(payload<0) {
            payload=p->npayloads++;
            v.payloads[payload]={origin,token};
            if(origin!=cfg.rank) p->remote_unique++;
          }
          v.tasks[p->ntasks++]={origin,token,slot,expert%(cfg.experts/cfg.ep),payload,0};
          if(origin==cfg.rank) p->local_pairs++; else p->remote_pairs++;
        }
      }
    }
    weave::Schedule s;
    if(cfg.fixed_c>=0) {
      s={cfg.fixed_c,cfg.fixed_k,cfg.fixed_steal,0.};
      if(p->npayloads<s.k) s.k=p->npayloads>0?p->npayloads:1;
      check(s.c>0 || p->remote_pairs==0);
    } else s=weave::choose_schedule(cfg.profile,cfg.nprofile,cfg.workers,p->local_pairs,
                                     p->remote_pairs,p->remote_unique,p->npayloads,cfg.h,cfg.i);
    check(s.c>=0 && s.c<cfg.workers && s.k>=1 && s.k<=MAX_K);
    p->c=s.c; p->k=s.k; p->steal=s.steal;
    for(int task=0;task<p->ntasks;++task) {
      int k=v.tasks[task].payload*s.k/p->npayloads;
      v.tasks[task].chunk=k; p->ends[k]++;
    }
    int start=0;
    for(int k=0;k<s.k;++k) { int count=p->ends[k]; p->starts[k]=p->next[k]=start; start+=count; p->ends[k]=start; }
    release(&p->planned,epoch);
  }
  if(threadIdx.x==0) wait_epoch(&p->planned,epoch,cfg.timeout);
  __syncthreads();
  bool comm=blockIdx.x<unsigned(p->c);
  // Dispatch is not chunked: pull each unique inbound token exactly once.
  if(comm || p->c==0) {
    while(true) {
      if(threadIdx.x==0) job=add_i(&p->dispatch_next,1);
      __syncthreads(); if(job>=p->npayloads) break;
      Payload token=v.payloads[job]; View from=cfg.peers[token.origin];
      for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x)
        v.cache[size_t(job)*cfg.h+col]=from.x[size_t(token.token)*cfg.h+col];
      __syncthreads(); __threadfence();
      if(threadIdx.x==0) { release(&v.cache_ready[job],epoch); add_i(&p->dispatch_done,1); }
      __syncthreads();
    }
    if(threadIdx.x==0) wait_count(&p->dispatch_done,p->npayloads,cfg.timeout);
    __syncthreads();
  }
  // One consecutive steal window immediately after dispatch.
  int budget=comm?p->steal:0x7fffffff;
  for(int stolen=0;stolen<budget;++stolen) {
    if(threadIdx.x==0) job=claim_task(p);
    __syncthreads(); if(job<0) break;
    compute_task(cfg,job,epoch,mid);
    if(comm && threadIdx.x==0) add_i(&p->actual_stolen,1);
  }
  // Chunk combine can overlap subsequent chunks' computation. Compute workers
  // join the tail after exhausting the shared compute queue.
  while(true) {
    if(threadIdx.x==0) job=add_i(&p->return_next,1);
    __syncthreads(); if(job>=p->ntasks) break;
    Task task=v.tasks[job];
    if(threadIdx.x==0) wait_count(&p->computed[task.chunk],p->ends[task.chunk]-p->starts[task.chunk],cfg.timeout);
    __syncthreads(); View to=cfg.peers[task.origin];
    size_t slot=size_t(task.token)*cfg.topk+task.slot;
    for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x)
      to.returned[slot*cfg.h+col]=v.task_out[size_t(job)*cfg.h+col];
    __syncthreads(); __threadfence_system();
    if(threadIdx.x==0) { release(&to.returned_ready[slot],epoch); add_i(&p->return_done,1); }
    __syncthreads();
  }
  // No CTA may wait for remote returns while this rank still owes a return.
  if(threadIdx.x==0) wait_count(&p->return_done,p->ntasks,cfg.timeout);
  __syncthreads();
  while(true) {
    if(threadIdx.x==0) job=add_i(&p->final_next,1);
    __syncthreads(); if(job>=p->n) break;
    if(threadIdx.x==0) for(int slot=0;slot<cfg.topk;++slot)
      wait_epoch(&v.returned_ready[size_t(job)*cfg.topk+slot],epoch,cfg.timeout);
    __syncthreads();
    for(int col=threadIdx.x;col<cfg.h;col+=blockDim.x) {
      float sum=0.;
      for(int slot=0;slot<cfg.topk;++slot) {
        size_t pos=size_t(job)*cfg.topk+slot;
        sum+=weave::routed_product(v.returned[pos*cfg.h+col],v.routing[pos],RoundBF16{});
      }
      v.output[size_t(job)*cfg.h+col]=__float2bfloat16_rn(sum);
    }
    __syncthreads();
  }
  __threadfence_system();
  if(threadIdx.x==0) {
    if(add_i(&p->ctas_done,1)==cfg.workers-1) release(&p->done,epoch);
    for(int peer=0;peer<cfg.ep;++peer) wait_epoch(&cfg.peers[peer].ctl->done,epoch,cfg.timeout);
  }
}

class Workspace : public std::enable_shared_from_this<Workspace> {
 public:
  int rank,ep,cap,h,i,topk,experts,device,workers;
  Layout layout;
  void* base=nullptr;
  void* pointers[MAX_EP]{};
  bool connected=false;
  Workspace(int rank_,int ep_,int cap_,int h_,int i_,int topk_,int experts_)
    : rank(rank_),ep(ep_),cap(cap_),h(h_),i(i_),topk(topk_),experts(experts_),layout(ep_,cap_,h_,topk_) {
    TORCH_CHECK(ep>=1 && ep<=MAX_EP && rank>=0 && rank<ep && cap>0 && cap<=8192);
    TORCH_CHECK(h>0 && h%32==0 && i>0 && i%32==0 && i<=8192 && h<=8192);
    TORCH_CHECK(topk>0 && topk<=16 && experts>=topk && experts%ep==0);
    CUDA_OK(cudaGetDevice(&device));
    TORCH_CHECK(device==rank,"functional IPC runtime requires CUDA device ordinal == EP rank");
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.major==9,"functional kernel targets Hopper SM90"); workers=prop.multiProcessorCount;
    CUDA_OK(cudaMalloc(&base,layout.bytes));
    auto stream=at::cuda::getCurrentCUDAStream(device).stream();
    CUDA_OK(cudaMemsetAsync(base,0,layout.bytes,stream)); CUDA_OK(cudaStreamSynchronize(stream));
    pointers[rank]=base;
  }
  ~Workspace() {
    if(!base) return;
    c10::cuda::CUDAGuard guard(device);
    for(int p=0;p<ep;++p) if(p!=rank && pointers[p]) cudaIpcCloseMemHandle(pointers[p]);
    cudaFree(base);
  }
  pybind11::bytes handle() {
    c10::cuda::CUDAGuard guard(device); cudaIpcMemHandle_t handle;
    CUDA_OK(cudaIpcGetMemHandle(&handle,base));
    return pybind11::bytes(reinterpret_cast<char*>(&handle),sizeof(handle));
  }
  void connect(const std::vector<std::string>& handles) {
    TORCH_CHECK(!connected && int(handles.size())==ep);
    c10::cuda::CUDAGuard guard(device);
    for(int p=0;p<ep;++p) {
      TORCH_CHECK(handles[p].size()==sizeof(cudaIpcMemHandle_t)); if(p==rank) continue;
      int access=0; CUDA_OK(cudaDeviceCanAccessPeer(&access,device,p)); TORCH_CHECK(access,"peer access unavailable");
      int atomics=0; CUDA_OK(cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,device,p));
      TORCH_CHECK(atomics,"system-scope native peer atomics required");
      cudaIpcMemHandle_t handle; memcpy(&handle,handles[p].data(),sizeof(handle));
      CUDA_OK(cudaIpcOpenMemHandle(&pointers[p],handle,cudaIpcMemLazyEnablePeerAccess));
    }
    connected=true;
  }
  torch::Tensor tensor(size_t offset,std::vector<int64_t> shape,torch::ScalarType dtype) {
    auto owner=shared_from_this();
    return torch::from_blob(static_cast<char*>(base)+offset,shape,[owner](void*){},torch::TensorOptions().dtype(dtype).device(torch::kCUDA,device));
  }
  torch::Tensor x() { return tensor(layout.x,{cap,h},torch::kBFloat16); }
  torch::Tensor ids() { return tensor(layout.ids,{cap,topk},torch::kInt32); }
  torch::Tensor routing() { return tensor(layout.routing,{cap,topk},torch::kFloat32); }
  torch::Tensor n() { return tensor(layout.control+offsetof(Control,n),{1},torch::kInt32); }
  torch::Tensor output() { return tensor(layout.output,{cap,h},torch::kBFloat16); }
  torch::Tensor schedule() { return tensor(layout.control+offsetof(Control,c),{3},torch::kInt32); }
  torch::Tensor work_counters() { return tensor(layout.control+offsetof(Control,actual_stolen),{2},torch::kInt32); }
  void run(const std::vector<torch::Tensor>& w,torch::Tensor profile,int c,int k,int steal,double timeout_seconds,float clamp,int tensor_core) {
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(connected && w.size()==6 && timeout_seconds>0 && clamp>0);
    TORCH_CHECK(c>=-1 && c<workers && k>=1 && k<=8 && steal>=0);
    TORCH_CHECK(tensor_core==0 || tensor_core==1,"unknown expert compute implementation");
    std::vector<std::vector<int64_t>> shapes={{experts/ep,i,h/2},{experts/ep,i,h/2},{experts/ep,h,i/2},
                                            {experts/ep,i,h/32},{experts/ep,i,h/32},{experts/ep,h,i/32}};
    for(int j=0;j<6;++j) TORCH_CHECK(w[j].is_cuda() && w[j].get_device()==device && w[j].is_contiguous() && w[j].scalar_type()==torch::kUInt8 && w[j].sizes().vec()==shapes[j],"bad raw MXFP4 weight tensor ",j);
    TORCH_CHECK(profile.is_cuda() && profile.get_device()==device && profile.is_contiguous() && profile.scalar_type()==torch::kFloat64 && profile.dim()==2 && profile.size(1)==6);
    TORCH_CHECK(c>=0 || profile.size(0)>0,"dynamic scheduling requires measured profile rows");
    int shared_bytes=i*2+(tensor_core?8192:0);
    int occupancy=0; CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,persistent,THREADS,shared_bytes));
    TORCH_CHECK(occupancy>=1,"persistent worker cannot reside on an SM");
    // cudaDeviceProp::clockRate was removed in CUDA 13; the driver attribute
    // returns the same kHz value.
    int clock_khz=0; CUDA_OK(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device));
    TORCH_CHECK(clock_khz>0,"device clock rate unavailable");
    Config cfg{};
    for(int p=0;p<ep;++p) cfg.peers[p]=layout.view(pointers[p]);
    cfg.rank=rank; cfg.ep=ep; cfg.capacity=cap; cfg.h=h; cfg.i=i; cfg.topk=topk; cfg.experts=experts; cfg.workers=workers;
    cfg.w1=w[0].data_ptr<unsigned char>(); cfg.w3=w[1].data_ptr<unsigned char>(); cfg.w2=w[2].data_ptr<unsigned char>();
    cfg.s1=w[3].data_ptr<unsigned char>(); cfg.s3=w[4].data_ptr<unsigned char>(); cfg.s2=w[5].data_ptr<unsigned char>();
    cfg.profile=profile.data_ptr<double>(); cfg.nprofile=profile.size(0); cfg.fixed_c=c; cfg.fixed_k=k; cfg.fixed_steal=steal;
    cfg.tensor_core=tensor_core;
    cfg.timeout=U64(timeout_seconds*clock_khz*1000.); cfg.clamp=clamp;
    auto stream=at::cuda::getCurrentCUDAStream(device).stream();
    prepare<<<1,1,0,stream>>>(cfg.peers[rank].ctl,cap);
    persistent<<<workers,THREADS,shared_bytes,stream>>>(cfg);
    CUDA_OK(cudaGetLastError());
  }
};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<Workspace,std::shared_ptr<Workspace>>(m,"Workspace")
    .def(pybind11::init<int,int,int,int,int,int,int>())
    .def("handle",&Workspace::handle).def("connect",&Workspace::connect)
    .def_property_readonly("x",&Workspace::x).def_property_readonly("ids",&Workspace::ids)
    .def_property_readonly("routing",&Workspace::routing).def_property_readonly("n",&Workspace::n)
    .def_property_readonly("output",&Workspace::output).def_property_readonly("schedule",&Workspace::schedule)
    .def_property_readonly("work_counters",&Workspace::work_counters)
    .def("run",&Workspace::run);
}
