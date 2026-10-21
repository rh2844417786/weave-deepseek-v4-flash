// Execute the actual bounded device planner with host threads and CTA barriers.
// This checks policy/grouping behavior; it is not CUDA execution evidence.
#include <atomic>
#include <barrier>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#define __CUDACC__
#define __host__
#define __device__
#define __global__
#define __shared__ static
struct Index { int x; };
thread_local Index threadIdx;
std::barrier cta_barrier(128);
void __syncthreads() { cta_barrier.arrive_and_wait(); }
int atomicAdd(int* p,int value) { return std::atomic_ref<int>(*p).fetch_add(value); }
int atomicExch(int* p,int value) { return std::atomic_ref<int>(*p).exchange(value); }
#include "native_cost.h"

int main() {
  int32_t sorted[192], experts[24], padded=192,control[3]={0,8,8};
  int total=192;
  int64_t plan[32]={};float table[432]={};
  const float feature[5]={192,24,24,1536,8};
  for(int i=0;i<192;++i)sorted[i]=i;
  for(int i=0;i<24;++i)experts[i]=-1;
  for(int candidate=0;candidate<9;++candidate)for(int a=0;a<6;++a) {
    float* row=table+candidate*48+a*8;
    if(a>0)for(int f=0;f<5;++f)row[f]=feature[f];
    row[5]=candidate==0 ? (a==0 ? 10 : 100) : candidate==8 ? (a==0 ? 100 : 10) : 200;
    row[6]=2;row[7]=3;
  }
  auto run=[&] {
    std::vector<std::thread> threads;
    for(int t=0;t<128;++t)threads.emplace_back([&,t] {
      threadIdx.x=t;
      weave_cost_plan_kernel(sorted,experts,&padded,192,24,8,total,plan,table,control);
    });
    for(auto &thread:threads)thread.join();
    for(auto value:plan)std::cout<<value<<' ';std::cout<<'\n';
  };
  run(); // Same capacities, remote-only actual grouping.
  for(int i=0;i<24;++i)experts[i]=i;
  run(); // Heavy local grouping chooses the other joint CTA pair.
  for(auto &value:table)value=0;
  control[0]=1;control[1]=4;control[2]=2;
  run(); // Fixed calibration mode works before any timing table exists.
  control[0]=0;
  run(); // Search must record invalid when timings are uncalibrated.
  control[0]=1;padded=193;
  run(); // Invalid padding must not read beyond the 192 allocated slots.
  padded=192;experts[0]=64;
  run(); // Invalid local expert index must not address past rows[64].
  total=6;padded=8;experts[0]=0;
  sorted[0]=6;for(int i=1;i<7;++i)sorted[i]=i-1;sorted[7]=6;
  run(); // Interior padding would make native Marlin read sentinel as a token.
  for(int i=0;i<6;++i)sorted[i]=i;sorted[6]=6;sorted[7]=6;
  run(); // The same number of valid slots with a valid prefix is accepted.
}
