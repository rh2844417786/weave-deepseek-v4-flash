#pragma once
#ifdef __CUDACC__
#define WEAVE_TRACE_HD __host__ __device__
#else
#define WEAVE_TRACE_HD
#endif

namespace weave { namespace trace {
constexpr int FIELDS=32;
enum Field {
  Begin,Planned,DispatchBegin,DispatchEnd,ComputeBegin,ComputeEnd,
  ReturnBegin,ReturnEnd,ReduceBegin,ReduceEnd,EndWaitBegin,End,
  DispatchWork,DispatchWait,ComputeWork,InputWait,ComputeQueue,
  ReturnWork,ReturnWait,ReturnQueue,ReduceWork,ReduceWait,EndWait,PlanWait,
  Tiles,Stolen,Returns,Rows,Total,PlanWork,Samples,Reserved
};
WEAVE_TRACE_HD inline int sample_slot(unsigned long long sequence,int stride,int capacity) {
  return sequence%stride ? -1 : int(sequence/stride%capacity);
}
// One stream-ordered writer per CTA. Absolute timestamps never enter totals.
WEAVE_TRACE_HD inline void accumulate(unsigned long long* totals,const unsigned long long* row) {
  for(int field=DispatchWork;field<=Samples;++field) totals[field]+=row[field];
}
}}
#undef WEAVE_TRACE_HD
