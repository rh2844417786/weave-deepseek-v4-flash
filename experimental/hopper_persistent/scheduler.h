#pragma once
#include <cfloat>
#include <cstdint>
#ifdef __CUDACC__
#define WEAVE_HD __host__ __device__
#else
#define WEAVE_HD
#endif

namespace weave {
struct Schedule { int c, k, steal; double seconds; };
// Hopper M64 work only. Six doubles per measured row: c,K,FLOP/s,bytes/s,efficiency,overlap.
WEAVE_HD inline Schedule choose_schedule(const double* rows, int count, int workers,
    int local_pairs, int remote_pairs, int remote_unique, int unique,
    int hidden, int intermediate, const int* groups_by_chunks) {
  if (workers<1 || hidden<=0 || hidden%128 || intermediate<=0 || intermediate%128 ||
      local_pairs<0 || remote_pairs<0 || remote_unique<0 || unique<0 || !groups_by_chunks)
    return {-1,1,0,DBL_MAX};
  if (local_pairs + remote_pairs == 0) return {0, 1, 0, 0.};
  Schedule best{-1, 1, 0, DBL_MAX};
  const double unit = 6. * hidden * intermediate;
  const double dispatch = double(remote_unique) * hidden * 2.;
  const double combine = double(remote_pairs) * hidden * 4.;
  for (int j = 0; j < count; ++j) {
    const double* r = rows + 6*j;
    bool finite = true;
    for (int q = 0; q < 6; ++q) finite = finite && r[q]==r[q] && r[q]<=DBL_MAX && r[q]>=-DBL_MAX;
    if (!finite || r[0]<0 || r[0]>=workers || r[1]<1 || r[1]>8) continue;
    const int c = int(r[0]), k = int(r[1]);
    if (r[0] != c || r[1] != k || c < 0 || c >= workers || k < 1 || k > 8 || k > unique) continue;
    if ((c == 0 && remote_pairs) || !(r[2] > 0.) || !(r[4] > 0. && r[4] <= 1.) || !(r[5] >= 0. && r[5] < 1.)) continue;
    if (!(r[3] >= 0.) || (c && r[3] == 0.)) continue;
    const double work = groups_by_chunks[k-1]*64.*unit;
    if (!(work>0.)) continue;
    const double comm = c ? (dispatch + combine) / r[3] : 0.;
    const double tail = c ? (1-r[5]) * combine / (r[3]*k) : 0.;
    const double comp = work / (r[2]*r[4]) + tail;
    const double total = comp > comm ? comp : comm;
    double residual = work - comm * r[2];
    if (residual < 0.) residual = 0.;
    const double steal_unit = 64.*unit / ((hidden+63)/64+(intermediate+63)/64);
    const double steal_estimate = c ? residual / (workers*steal_unit) : 0.;
    const int steal = steal_estimate > 2147483647. ? 2147483647 : int(steal_estimate);
    if (total < best.seconds || (total == best.seconds && (k < best.k || (k == best.k && c < best.c))))
      best = {c, k, steal, total};
  }
  return best;
}
}

#undef WEAVE_HD
