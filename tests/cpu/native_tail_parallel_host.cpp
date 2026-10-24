#include <cassert>
#include <random>
#include <vector>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
// Execute the production cooperative planner with host threads/barriers. This
// checks its synchronization/data partition, not NVCC or CUDA memory semantics.
struct Dim {unsigned x;};thread_local Dim threadIdx;Dim blockDim{8};
struct Barrier {std::mutex mutex;std::condition_variable cv;unsigned arrived=0,generation=0;
 void wait(){std::unique_lock<std::mutex>lock(mutex);unsigned before=generation;
  if(++arrived==blockDim.x){arrived=0;++generation;cv.notify_all();}
  else cv.wait(lock,[&]{return generation!=before;});
 }
} barrier;
#define __CUDACC__
#define __host__
#define __device__
#define __shared__ static
#define __syncthreads() barrier.wait()
#include "experimental/native_tail/cost_model.h"
#include "experimental/native_tail/task_schedule.h"
int main(){
  using namespace weave::native_tail;
  for(int tokens: {0,1,7,32,96})for(int chunks:{1,2,4})for(int threads:{128,256}){
    int n=tokens*4096/chunks, tasks=finalize_task_count(tokens,4096,chunks,threads);
    std::vector<int> visits(n,0);
    for(int task=tasks-1;task>=0;--task){
      auto range=finalize_task_range(task,n,threads);
      assert(range.begin<=range.end&&range.end<=n);
      for(int i=range.begin;i<range.end;++i)visits[i]++;
    }
    for(int count:visits)assert(count==1);
    assert(tasks>=1);
  }
  for(int workers:{2,3,132,396})for(int comm:{0,1,2,8})if(comm<workers)
    for(int chunks:{1,2,4})for(bool done:{false,true}){
      int active=0;for(int w=0;w<workers;++w)active+=communication_worker_eligible(w,comm,chunks,done);
      assert(active==std::min(workers,comm+(done?5*chunks:0)));
    }
  std::mt19937 gen(773);
  for(int repeat=0;repeat<120;++repeat){
    TailPlan plans[kCostMaxPlans];int count=1;plans[0]={0,1,0,1};
    int max_comm=repeat%3==0?1:repeat%3==1?2:8;
    for(int chunks:{1,2,4})for(int comm:{1,2,4,8})if(comm<=max_comm)
      for(int steal=0;steal<(comm==1?1:2);++steal)plans[count++]={comm,chunks,steal,0};
    std::vector<double> table(4*count*kCostAnchors*kCostRow);
    int anchors[4*kCostMaxPlans];double features[20];TailCurve curves[4*kCostMaxPlans];
    for(int i=0;i<20;++i)features[i]=gen()%10;
    for(int i=0;i<4*count;++i){anchors[i]=2+gen()%3;
      for(int j=0;j<kCostAnchors*kCostRow;++j)table[i*kCostAnchors*kCostRow+j]=repeat%4==0?1.:double(gen()%1000)/7.;}
    int reference[4];double best=select_tail_plan(table.data(),anchors,features,plans,count,192,curves,reference);
    std::fill(std::begin(curves),std::end(curves),TailCurve{});
    int parallel_chosen[4]={-1,-1,-1,-1};double observed[8]={};std::vector<std::thread>lanes;
    for(unsigned lane=0;lane<blockDim.x;++lane)lanes.emplace_back([&,lane]{threadIdx.x=lane;
      observed[lane]=select_tail_plan_parallel(table.data(),anchors,features,plans,count,192,curves,parallel_chosen);});
    for(auto&lane:lanes)lane.join();
    for(double cost:observed)assert(cost==best);
    for(int r=0;r<4;++r)assert(parallel_chosen[r]==reference[r]);
    int shortlist[24],lengths[12];
    for(int c=0;c<3;++c)for(int r=0;r<4;++r)
      lengths[c*4+r]=shortlist_tail_plans(curves,plans,count,r,1<<c,shortlist+(c*4+r)*2);
    int chosen[4]={0,0,0,0};double got=1.e300;int valid=0;
    // Reverse order exercises deterministic lexicographic tie handling.
    for(int candidate=count+48-1;candidate>=0;--candidate){int ids[4];
      if(!tail_candidate_ids(count,shortlist,lengths,candidate,ids))continue;
      ++valid;consider_plan(curves,plans,count,ids,chosen,got);
      for(int r=0;r<4;++r)assert(plans[ids[r]].chunks==plans[ids[0]].chunks);
    }
    assert(valid>=count&&got==best);
    for(int r=0;r<4;++r)assert(chosen[r]==reference[r]);
  }
}
