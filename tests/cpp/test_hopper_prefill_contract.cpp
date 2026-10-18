#include "math_contract.h"
#include <cassert>
#include <random>
#include <vector>
using namespace weave::hopper_prefill;
int main() {
  assert(valid_dimensions(3,4,8192,4096,2048,256,6));
  assert(!valid_dimensions(4,4,8192,4096,2048,256,6));
  assert(!valid_dimensions(0,4,8193,4096,2048,256,6));
  assert(!valid_dimensions(0,4,1,129,2048,256,6));
  assert(tile_capacity(8192,6,64)==832);
  assert(tile_launch_capacity(1,6,64)==7);
  std::mt19937 gen(123);
  for(int batch: {1,7,33,65,129,8035,8192}) {
    // Model contiguous thread ranges used by the real GPU grouping: every
    // token visited exactly once and ranges concatenate in stable order.
    int next=0;
    for(int thread=0;thread<256;++thread) {
      assert(token_range_start(batch,thread)==next);
      for(int t=token_range_start(batch,thread);t<token_range_start(batch,thread+1);++t) assert(t==next++);
    }
    assert(next==batch);
    for(int trial=0;trial<100;++trial) {
      std::vector<int> counts(64,0);
      for(int t=0;t<batch;++t) {
        int start=gen()%64;
        for(int k=0;k<6;++k) ++counts[(start+k)%64];
      }
      int tiles=0;
      for(int count:counts) { assert(count<=batch); tiles+=(count+63)/64; }
      assert(tiles<=tile_launch_capacity(batch,6,64));
      assert(tiles<=tile_capacity(8192,6,64));
    }
  }
}
