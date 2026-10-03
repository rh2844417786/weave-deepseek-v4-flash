#include "tile_scheduler.h"
#include <cassert>
#include <set>

int main() {
  using namespace weave;
  assert(tile_group_capacity(97, 3, 2) == 13);
  assert(tile_columns(32)==1 && tile_columns(2048)==32 && tile_columns(4096)==64);
  TileState state[2]{};
  auto load=[](int* p){return *p;};
  auto add=[](int* p,int n){int old=*p;*p+=n;return old;};
  std::set<int> seen_gate;
  for(int j=0;j<4;++j) {
    auto q=claim_tile(state,2,2,3,load,add);
    assert(q.stage==0 && q.group>=0);
    assert(seen_gate.insert(q.group*2+q.column).second);
  }
  assert(claim_tile(state,2,2,3,load,add).group==-2); // Work is inflight.
  state[1].gate_done=2;
  for(int j=0;j<3;++j) {
    auto q=claim_tile(state,2,2,3,load,add);
    assert(q.group==1 && q.stage==1 && q.column==j);
  }
  assert(claim_tile(state,2,2,3,load,add).group==-2);
  state[1].down_done=3;
  state[0].gate_done=2;
  for(int j=0;j<3;++j) {
    auto q=claim_tile(state,2,2,3,load,add);
    assert(q.group==0 && q.stage==1 && q.column==j);
  }
  state[0].down_done=3;
  assert(claim_tile(state,2,2,3,load,add).group==-1);
  assert(claim_tile(state,0,2,3,load,add).group==-1);
  TileState roundrobin[2]{};
  int cursor=0;
  std::set<int> all;
  for(int j=0;j<10;++j) {
    auto q=claim_tile(roundrobin,2,2,3,load,add,&cursor);
    assert(q.group>=0);
    assert(all.insert(q.group*10+q.stage*3+q.column).second);
    if(q.stage==0) ++roundrobin[q.group].gate_done;
    else ++roundrobin[q.group].down_done;
  }
  assert(claim_tile(roundrobin,2,2,3,load,add,&cursor).group==-1);
}
