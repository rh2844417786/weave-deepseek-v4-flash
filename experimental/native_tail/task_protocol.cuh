#pragma once
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace weave::native_tail {
using U64 = unsigned long long;
constexpr int kRanks = 4, kSlots = 2, kMaxChunks = 4, kMaxTasks = 20;
constexpr int kProtocolVersion = 1, kThreads = 128;
constexpr U64 kCanary = 0x613cd2e9a78bf045ULL;
enum AbortCode : U64 { kOk=0, kTimeout=1, kIdentity=2, kNonfinite=3,
                       kReuse=4, kCancelled=5, kSequenceOverflow=6 };

// All remote state is read-only. Every mutable word belongs to its allocating rank.
// epoch/ready/reduced/ack/done are monotonically tagged; never bare reusable flags.
struct alignas(128) SlotControl {
  U64 epoch, session, layer, tokens, hidden, chunks, dtype, version;
  U64 ready[kMaxChunks], reduced[kMaxChunks], ack[kMaxChunks][kRanks], done;
  unsigned claimed[kMaxTasks], executed[kMaxTasks], smid[kMaxTasks];
  unsigned completed, nonfinite;
};
// Optional GPU planner exchange owned by executor; standalone transport ignores it.
struct alignas(128) PlannerMailbox {
  U64 workload_epoch, plan_epoch, plan_ack;
  double workload[8];
  int plan[16];
};
struct alignas(128) Control {
  U64 sequence, abort_code;
  PlannerMailbox planner;
  SlotControl slots[kSlots];
};
struct Layout {
  int capacity, hidden;
  size_t plane_elements, plane_bytes, slot_bytes, bytes;
  __host__ __device__ Layout(int capacity_, int hidden_): capacity(capacity_), hidden(hidden_) {
    plane_elements=size_t(capacity)*hidden;
    plane_bytes=((plane_elements*sizeof(__nv_bfloat16)+127)/128)*128;
    slot_bytes=2*plane_bytes+256;
    bytes=sizeof(Control)+kSlots*slot_bytes;
  }
  __host__ __device__ __nv_bfloat16* partial(void* base, int slot) const {
    return reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(base)+sizeof(Control)+slot*slot_bytes+128);
  }
  __host__ __device__ __nv_bfloat16* reduced(void* base, int slot) const {
    return reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(partial(base,slot))+plane_bytes);
  }
  __host__ __device__ U64* guard(void* base, int slot, bool after) const {
    return reinterpret_cast<U64*>(reinterpret_cast<char*>(base)+sizeof(Control)+slot*slot_bytes+(after?128+2*plane_bytes:0));
  }
};
struct Invocation { U64 sequence; int slot; };
struct TransportConfig {
  void* peers[kRanks];
  Layout layout;
  int rank, tokens, chunks;
  U64 session, layer, timeout_cycles, delay_cycles;
  __nv_bfloat16* output;
  // Controlled GPU cancellation injection; 0 disables it. Host watchdog owns process cancellation.
  U64 cancel_after_cycles;
  bool diagnostics;
  TransportConfig():layout(0,0), rank(0),tokens(0),chunks(1),session(0),layer(0),
    timeout_cycles(0),delay_cycles(0),output(nullptr),cancel_after_cycles(0),diagnostics(true) {
    for(int r=0;r<kRanks;++r) peers[r]=nullptr;
  }
};
__device__ inline U64 sys_load(U64* p) {
  return cuda::atomic_ref<U64,cuda::thread_scope_system>(*p).load(cuda::memory_order_acquire);
}
__device__ inline void sys_store(U64* p,U64 value) {
  cuda::atomic_ref<U64,cuda::thread_scope_system>(*p).store(value,cuda::memory_order_release);
}
__device__ inline unsigned dev_load(unsigned* p) {
  return cuda::atomic_ref<unsigned,cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}
__device__ inline void dev_store(unsigned* p,unsigned value) {
  cuda::atomic_ref<unsigned,cuda::thread_scope_device>(*p).store(value,cuda::memory_order_release);
}
__device__ inline Control* control(const TransportConfig& cfg,int rank) {
  return static_cast<Control*>(cfg.peers[rank]);
}
__device__ inline SlotControl* slot_control(const TransportConfig& cfg,int rank,Invocation inv) {
  return &control(cfg,rank)->slots[inv.slot];
}
__device__ inline void abort_local(const TransportConfig& cfg,U64 code) {
  auto ref=cuda::atomic_ref<U64,cuda::thread_scope_system>(control(cfg,cfg.rank)->abort_code);
  U64 zero=0; ref.compare_exchange_strong(zero,code,cuda::memory_order_release,cuda::memory_order_relaxed);
}
__device__ inline bool protocol_aborted(const TransportConfig& cfg,U64 start) {
  if(cfg.cancel_after_cycles && clock64()-start>cfg.cancel_after_cycles)
    abort_local(cfg,kCancelled);
  if(clock64()-start>cfg.timeout_cycles) abort_local(cfg,kTimeout);
  for(int r=0;r<kRanks;++r) if(sys_load(&control(cfg,r)->abort_code)) {
    abort_local(cfg,sys_load(&control(cfg,r)->abort_code)); return true;
  }
  return false;
}
// Call ONCE from grid leader, then cooperative grid.sync() before any local access.
// A previous invocation must finish transport_complete() before returning/reusing.
__device__ inline Invocation prepare_invocation(const TransportConfig& cfg) {
  Control* local=control(cfg,cfg.rank);
  U64 sequence=local->sequence+1;
  if(sequence==0) {abort_local(cfg,kSequenceOverflow); sequence=1;}
  Invocation inv{sequence,int((sequence-1)%kSlots)};
  SlotControl* slot=slot_control(cfg,cfg.rank,inv);
  U64 old=sys_load(&slot->epoch);
  if(old) for(int r=0;r<kRanks;++r)
    if(sys_load(&slot_control(cfg,r,inv)->done)<old) abort_local(cfg,kReuse);
  // Only local task state is reset. Tagged peer publication words persist.
  for(int t=0;t<kMaxTasks;++t) {slot->claimed[t]=0;slot->executed[t]=0;slot->smid[t]=~0u;}
  slot->completed=0;slot->nonfinite=0;
  slot->session=cfg.session;slot->layer=cfg.layer;slot->tokens=cfg.tokens;
  slot->hidden=cfg.layout.hidden;slot->chunks=cfg.chunks;slot->dtype=1;slot->version=kProtocolVersion;
  local->sequence=sequence;
  sys_store(&slot->epoch,sequence);
  return inv;
}
__device__ inline Invocation current_invocation(const TransportConfig& cfg) {
  U64 sequence=control(cfg,cfg.rank)->sequence;
  return {sequence,int((sequence-1)%kSlots)};
}
// Nonblocking: caller can perform independent down work while a peer is absent.
__device__ inline bool protocol_headers_ready(const TransportConfig& cfg,Invocation inv) {
  for(int r=0;r<kRanks;++r) {
    SlotControl* s=slot_control(cfg,r,inv); U64 epoch=sys_load(&s->epoch);
    if(epoch<inv.sequence) return false;
    if(epoch!=inv.sequence || s->session!=cfg.session || s->layer!=cfg.layer ||
       s->tokens!=U64(cfg.tokens) || s->hidden!=U64(cfg.layout.hidden) ||
       s->chunks!=U64(cfg.chunks) || s->dtype!=1 || s->version!=kProtocolVersion) {
      abort_local(cfg,kIdentity); return false;
    }
  }
  return true;
}
// Every payload writer must call its own fence before the CTA leader publishes.
// For many-CTA finalize: every producing lane first system-fences its payload;
// each CTA then releases completion. The last CTA acquires that completion chain
// before calling here to publish readiness for the whole chunk.
__device__ inline void publish_local_chunk(const TransportConfig& cfg,Invocation inv,int chunk) {
  __threadfence_system(); __syncthreads();
  if(threadIdx.x==0) sys_store(&slot_control(cfg,cfg.rank,inv)->ready[chunk],inv.sequence);
  __syncthreads();
}
__device__ inline size_t payload_index(const TransportConfig& cfg,int chunk,int owner,int token,int x) {
  int width=cfg.layout.hidden/(cfg.chunks*kRanks);
  return ((size_t(chunk)*kRanks+owner)*cfg.layout.capacity+token)*width+x;
}
__device__ inline size_t output_index(const TransportConfig& cfg,int chunk,int owner,int token,int x) {
  int width=cfg.layout.hidden/(cfg.chunks*kRanks);
  return size_t(token)*cfg.layout.hidden+(chunk*kRanks+owner)*width+x;
}
} // namespace weave::native_tail
