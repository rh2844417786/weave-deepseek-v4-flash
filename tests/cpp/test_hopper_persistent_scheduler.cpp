#include <cassert>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>
#include <atomic>
#include "../../experimental/hopper_persistent/scheduler.h"
#include "../../experimental/hopper_persistent/tile_scheduler.h"
struct Load { int operator()(int* p) const { return *p; } };
struct Add { int operator()(int* p,int v) const { int old=*p; *p+=v; return old; } };
struct Ready { bool* ready; bool operator()(int group) const { return ready[group]; } };
struct AtomicLoad { int operator()(int* p) const { return __atomic_load_n(p,__ATOMIC_ACQUIRE); } };
struct AtomicAdd { int operator()(int* p,int v) const { return __atomic_fetch_add(p,v,__ATOMIC_ACQ_REL); } };
struct AtomicReady { int* ready; bool operator()(int group) const { return __atomic_load_n(ready+group,__ATOMIC_ACQUIRE)!=0; } };
void concurrent_claims() {
  constexpr int groups=17,gate=4,down=3;
  weave::TileState states[groups]{};int ready[groups]{};
  int gate_data[groups][gate]{};int visits[groups][gate+down]{};
  for(int q=1;q<groups;++q) ready[q]=1;
  std::atomic<int> later_claims{0};
  std::vector<std::thread> workers;
  for(int worker=0;worker<8;++worker) workers.emplace_back([&] {
    for(;;) {
      auto claim=weave::claim_tile(states,groups,gate,down,AtomicLoad{},AtomicAdd{},AtomicReady{ready});
      if(claim.group==-1) break;
      if(claim.group==-2) { std::this_thread::yield();continue; }
      if(claim.group>0 && later_claims.fetch_add(1)==3) __atomic_store_n(ready,1,__ATOMIC_RELEASE);
      const int group=claim.group,column=claim.column;
      assert(__atomic_fetch_add(&visits[group][claim.stage?gate+column:column],1,__ATOMIC_RELAXED)==0);
      if(claim.stage==0) {
        gate_data[group][column]=column+1;
        std::this_thread::yield();
        __atomic_fetch_add(&states[group].gate_done,1,__ATOMIC_ACQ_REL);
      } else {
        for(int k=0;k<gate;++k) assert(gate_data[group][k]==k+1);
        __atomic_fetch_add(&states[group].down_done,1,__ATOMIC_ACQ_REL);
      }
    }
  });
  for(auto& worker:workers) worker.join();
  for(auto& group:visits) for(int count:group) assert(count==1);
}
int main() {
  static_assert(weave::TILE_ROWS==64);
  int groups[8]={2,3,4,5,6,7,8,9};
  double profile[]={0,1,1e12,0,1,0, 1,2,1e12,1e12,1,0};
  auto no_work=weave::choose_schedule(profile,2,8,0,0,0,0,128,128,groups);
  assert(no_work.c==0 && no_work.k==1 && no_work.steal==0);
  auto local=weave::choose_schedule(profile,2,8,1,0,0,4,128,128,groups);
  assert(local.c==0 && local.k==1);
  assert(std::abs(local.seconds-2.*64*6*128*128/1e12)<1e-15);
  auto remote=weave::choose_schedule(profile,2,8,0,4,2,4,128,128,groups);
  assert(remote.c==1 && remote.k==2 && remote.steal>=0);
  assert(weave::choose_schedule(profile,1,8,0,1,1,1,128,128,groups).c==-1);
  double invalid[]={1,1,std::numeric_limits<double>::quiet_NaN(),1,1,0};
  assert(weave::choose_schedule(invalid,1,8,0,1,1,1,128,128,groups).c==-1);
  weave::TileState states[3]{}; bool ready[]={false,true,true};
  auto claim=weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready});
  assert(claim.group==1 && claim.stage==0 && claim.column==0);
  ready[0]=true;
  claim=weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready});
  assert(claim.group==0 && claim.stage==0 && claim.column==0);
  claim=weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready});
  assert(claim.group==0 && claim.column==1);
  states[0].gate_done=2;
  claim=weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready});
  assert(claim.group==0 && claim.stage==1 && claim.column==0);
  // A down stage can only become visible after every gate tile publishes.
  states[0]={2,1,0,0}; states[1]={2,1,0,0}; states[2]={2,1,0,0};
  assert(weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready}).group==-2);
  for(auto& s:states) s={2,2,2,2};
  assert(weave::claim_tile(states,3,2,2,Load{},Add{},Ready{ready}).group==-1);
  assert(weave::tile_group_capacity(65,2,2)>=2+4);
  concurrent_claims();
}
