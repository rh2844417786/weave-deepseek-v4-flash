#include "math_contract.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
using namespace weave::hopper;
int main() {
  // Sign, half-unit magnitude, and low/high nibble order must be exact.
  const uint8_t expected[16] = {0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,
                               0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc};
  for (int n=0;n<16;++n) assert(nibble_to_e4m3(n)==expected[n]);
  assert(packed_to_e4m3(0xe3,0)==0x3c);
  assert(packed_to_e4m3(0xe3,1)==0xc8);
  assert(b128_offset(0,0)==0 && b128_offset(1,0)==144);
  assert(b128_offset(7,96)==912 && b128_offset(8,0)==1024);
  std::vector<bool> seen(8192,false);
  for(int r=0;r<64;++r) for(int k=0;k<128;++k) {
    auto offset=b128_offset(r,k);
    assert(offset<8192 && !seen[offset]); seen[offset]=true;
  }
  assert(activation_scale(0.f)==1.f);
  assert(activation_scale(448.f)==1.f);
  assert(activation_scale(224.f)==.5f);
  assert(valid_dimensions(1,4,32,4096,2048,256,6));
  assert(!valid_dimensions(4,4,32,4096,2048,256,6));
  assert(!valid_dimensions(0,4,33,4096,2048,256,6));
  assert(!valid_dimensions(0,4,32,4097,2048,256,6));
  assert(active_capacity(32,6,64)==64 && active_capacity(1,6,64)==6);
  int counts[4],active[4],slots[4*3];
  const int64_t ids[]={4,1,7,5,4,0,7,4,2};
  int count=plan_routes(ids,3,3,8,1,2,3,counts,active,slots);
  assert(count==3 && active[0]==0 && active[1]==1 && active[2]==3);
  assert(counts[0]==3 && counts[1]==1 && counts[2]==0 && counts[3]==2);
  assert(slots[0]==0 && slots[1]==4 && slots[2]==7);
  assert(slots[3]==3 && slots[9]==2 && slots[10]==6 && slots[11]==-1);
  const int64_t duplicate[]={4,4,1};
  assert(plan_routes(duplicate,1,3,8,1,2,3,counts,active,slots)==-1);
  const int64_t huge[]={int64_t(1)<<40,1,7};
  assert(plan_routes(huge,1,3,8,1,2,3,counts,active,slots)==-1);
  const int64_t remote[]={0,1,2};
  assert(plan_routes(remote,1,3,8,1,2,3,counts,active,slots)==0);
  const int64_t full[]={4,5,7,4,5,7,4,5,7};
  assert(plan_routes(full,3,3,8,1,2,3,counts,active,slots)==3);
  assert(counts[0]==3 && counts[1]==3 && counts[3]==3);
}
