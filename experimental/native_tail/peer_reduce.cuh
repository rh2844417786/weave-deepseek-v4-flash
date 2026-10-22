#pragma once
#include "task_protocol.cuh"
#include <math.h>

namespace weave::native_tail {
// CTA-consistent, bounded try-work. No claimed task waits for an unready dependency.
// Task ids [0,C) = OWNER_REDUCE, [C,5C) = FETCH_RESULT(chunk,owner).
// OWNER_REDUCE fuses pull of four source partials with fixed-rank FP32 accumulation.
__device__ inline int transport_try_work(const TransportConfig& cfg,Invocation inv,U64* readiness_clock=nullptr) {
  __shared__ int selected;
  SlotControl* local=slot_control(cfg,cfg.rank,inv);
  if(threadIdx.x==0) {
    selected=-1;
    for(int c=0;c<cfg.chunks && selected<0;++c) {
      bool ready=true;
      for(int r=0;r<kRanks;++r) ready &= sys_load(&slot_control(cfg,r,inv)->ready[c])==inv.sequence;
      if(ready && atomicCAS(&local->claimed[c],0u,1u)==0) selected=c;
    }
    for(int c=0;c<cfg.chunks && selected<0;++c) for(int owner=0;owner<kRanks && selected<0;++owner) {
      int task=cfg.chunks+c*kRanks+owner;
      if(sys_load(&slot_control(cfg,owner,inv)->reduced[c])==inv.sequence &&
         atomicCAS(&local->claimed[task],0u,1u)==0) selected=task;
    }
  }
  __syncthreads();
  int task=selected;
  if(task<0) return -1;
  // Diagnostic timestamp is recorded on the successful ready claim, including
  // the race where ready appeared after the executor's preceding nonblocking poll.
  if(readiness_clock && threadIdx.x==0){U64 time;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time));
    atomicCAS(readiness_clock,0ULL,time);}
  __syncthreads();
  int c=task<cfg.chunks?task:(task-cfg.chunks)/kRanks;
  int owner=task<cfg.chunks?cfg.rank:(task-cfg.chunks)%kRanks;
  int width=cfg.layout.hidden/(cfg.chunks*kRanks);
  size_t n=size_t(cfg.tokens)*width;
  if(task<cfg.chunks) {
    // Every reader acquires the publication before its peer payload reads.
    for(int r=0;r<kRanks;++r) (void)sys_load(&slot_control(cfg,r,inv)->ready[c]);
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x) {
      size_t p=payload_index(cfg,c,owner,int(i/width),int(i%width));
      float sum=0;
      for(int r=0;r<kRanks;++r) sum+=__bfloat162float(cfg.layout.partial(cfg.peers[r],inv.slot)[p]);
      __nv_bfloat16 value=__float2bfloat16_rn(sum);
      if(!isfinite(sum)||!isfinite(__bfloat162float(value))) {
        atomicAdd(&local->nonfinite,1u);abort_local(cfg,kNonfinite);
      }
      cfg.layout.reduced(cfg.peers[cfg.rank],inv.slot)[p]=value;
    }
    __threadfence_system(); __syncthreads();
    if(threadIdx.x==0) sys_store(&local->reduced[c],inv.sequence);
  } else {
    (void)sys_load(&slot_control(cfg,owner,inv)->reduced[c]);
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x) {
      int token=int(i/width),x=int(i%width);
      cfg.output[output_index(cfg,c,owner,token,x)]=
        cfg.layout.reduced(cfg.peers[owner],inv.slot)[payload_index(cfg,c,owner,token,x)];
    }
    // Consumer acknowledgement happens only after every read finishes.
    __threadfence_system(); __syncthreads();
    if(threadIdx.x==0) sys_store(&local->ack[c][owner],inv.sequence);
  }
  __syncthreads();
  if(threadIdx.x==0) {
    if(cfg.diagnostics) {
      unsigned sm; asm("mov.u32 %0, %%smid;":"=r"(sm));
      local->smid[task]=sm;atomicAdd(&local->executed[task],1u);
    }
    cuda::atomic_ref<unsigned,cuda::thread_scope_device>(local->completed).fetch_add(1,cuda::memory_order_acq_rel);
  }
  __syncthreads();return task;
}
// Leader-only nonblocking completion. Caller must keep progressing/checking abort.
// In addition to all local gathers, all peers must have consumed OUR owner shards.
__device__ inline bool transport_complete(const TransportConfig& cfg,Invocation inv) {
  SlotControl* local=slot_control(cfg,cfg.rank,inv);
  if(dev_load(&local->completed)!=unsigned(cfg.chunks*5)) return false;
  for(int c=0;c<cfg.chunks;++c) for(int r=0;r<kRanks;++r)
    if(sys_load(&slot_control(cfg,r,inv)->ack[c][cfg.rank])!=inv.sequence) return false;
  sys_store(&local->done,inv.sequence);
  for(int r=0;r<kRanks;++r)
    if(sys_load(&slot_control(cfg,r,inv)->done)<inv.sequence) return false;
  return true;
}
} // namespace weave::native_tail
