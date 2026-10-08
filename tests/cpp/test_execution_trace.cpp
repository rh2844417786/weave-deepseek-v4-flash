#include <cassert>
#include "execution_trace.h"
int main() {
  using namespace weave::trace;
  assert(sample_slot(0,2,2)==0);
  assert(sample_slot(1,2,2)==-1);
  assert(sample_slot(2,2,2)==1);
  assert(sample_slot(4,2,2)==0);
  unsigned long long row[FIELDS]{},total[FIELDS]{};
  row[Begin]=100; row[End]=250; row[ComputeWork]=80;
  row[Total]=150; row[Samples]=1;
  accumulate(total,row); accumulate(total,row);
  assert(total[Begin]==0 && total[End]==0);
  assert(total[ComputeWork]==160 && total[Total]==300 && total[Samples]==2);
}
