#include "scheduler.h"
#include <cassert>
#include <cmath>
int main() {
  double rows[]={1,1,120,20,1,.5, 1,2,120,20,.9,.5, 1,4,120,20,.5,.5};
  auto p=weave::choose_schedule(rows,3,4,6,4,2,10,2,2);
  assert(p.c==1 && p.k==2 && p.steal==0);
  assert(std::abs(p.seconds-(240./108.+.4))<1e-12);
  double invalid[]={0,1,24,0,1,0};
  assert(weave::choose_schedule(invalid,1,4,0,1,1,1,2,2).c==-1);
  auto local=weave::choose_schedule(invalid,1,4,1,0,0,1,2,2);
  assert(local.c==0 && local.seconds==1.);
  double tiled[]={1,1,1e9,1e12,1,0, 1,2,1.5e9,1e12,1,0};
  int groups[]={1,2,0,0,0,0,0,0};
  assert(weave::choose_schedule(tiled,2,4,7,1,1,8,32,32).k==2);
  // Splitting a sparse group into two padded M16 tiles reverses that choice.
  assert(weave::choose_schedule(tiled,2,4,7,1,1,8,32,32,groups).k==1);
}
