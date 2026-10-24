#pragma once
#ifdef __CUDACC__
#define NT_SCHED_HD __host__ __device__
#else
#define NT_SCHED_HD
#endif
namespace weave::native_tail {
// Small contiguous slices retain the original per-element reduction order.
constexpr int kFinalizeElementsPerThread=8;
struct FinalizeRange {int begin,end;};
NT_SCHED_HD inline int finalize_task_count(int tokens,int hidden,int chunks,int threads){
  int elements=tokens*(hidden/chunks),tile=threads*kFinalizeElementsPerThread;
  return elements?(elements+tile-1)/tile:1;
}
NT_SCHED_HD inline FinalizeRange finalize_task_range(int task,int elements,int threads){
  int begin=task*threads*kFinalizeElementsPerThread;
  int end=begin+threads*kFinalizeElementsPerThread;
  return {begin<elements?begin:elements,end<elements?end:elements};
}
NT_SCHED_HD inline bool communication_worker_eligible(int worker,int comm,int chunks,bool finalized){
  // At most one helper per actual peer task after compute finishes. Idle compute
  // CTAs must not all issue system-scope polling traffic on the same mailboxes.
  return worker<comm||(finalized&&worker<comm+5*chunks);
}
} // namespace weave::native_tail
