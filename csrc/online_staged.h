#pragma once
#include "tile_scheduler.h"

#ifdef __CUDACC__
#define WEAVE_ONLINE_HD __host__ __device__
#else
#define WEAVE_ONLINE_HD
#endif

namespace weave {
// Check wide native-router IDs before narrowing into preallocated int32 work.
template<class Id>
WEAVE_ONLINE_HD inline bool online_copy_ids(const Id* source, int* destination,
                                           int count, int experts) {
  for (int pos = 0; pos < count; ++pos) {
    if (source[pos] < 0 || source[pos] >= experts) return false;
    destination[pos] = int(source[pos]);
  }
  return true;
}

// Each encountered expert consumes a first group. Every extra group requires
// TILE_ROWS additional routes beyond those first encounters. This bound also
// covers overflow groups when B > 16; bounding by local_experts alone does not.
WEAVE_ONLINE_HD inline int online_group_capacity(int batch, int topk, int local_experts) {
  const int routes = batch * topk;
  const int distinct = routes < local_experts ? routes : local_experts;
  const int bound = distinct + (routes - distinct) / TILE_ROWS;
  return bound > 0 ? bound : 1;
}

// A single planner thread deliberately preserves P1's first-encounter order.
// All arrays are supplied by the caller; no allocation or host routing reads.
// Return -1 for configuration, -2 for routes, -3 for insufficient capacity.
WEAVE_ONLINE_HD inline int online_build_groups(const int* ids, int batch, int topk,
    int rank, int ep, int experts, int capacity, int* meta, int* slots) {
  if (batch < 1 || batch > 32 || topk < 1 || topk > 16 || ep < 1 || ep > 4 ||
      rank < 0 || rank >= ep || experts < topk || experts % ep || capacity < 1)
    return -1;
  for (int group = 0; group < capacity; ++group) {
    meta[group * 2] = -1; meta[group * 2 + 1] = 0;
    for (int row = 0; row < TILE_ROWS; ++row) slots[group * TILE_ROWS + row] = -1;
  }
  int count = 0;
  const int local_experts = experts / ep;
  for (int pos = 0; pos < batch * topk; ++pos) {
    const int expert = ids[pos];
    if (expert < 0 || expert >= experts) return -2;
    for (int prev = (pos / topk) * topk; prev < pos; ++prev)
      if (ids[prev] == expert) return -2;
    if (expert / local_experts != rank) continue;
    const int local = expert % local_experts;
    int group = -1;
    for (int candidate = count - 1; candidate >= 0; --candidate) {
      if (meta[candidate * 2] == local) {
        if (meta[candidate * 2 + 1] < TILE_ROWS) group = candidate;
        break;
      }
    }
    if (group < 0) {
      if (count >= capacity) return -3;
      group = count++;
      meta[group * 2] = local;
    }
    const int row = meta[group * 2 + 1]++;
    slots[group * TILE_ROWS + row] = pos;
  }
  return count;
}

WEAVE_ONLINE_HD inline bool online_origin_row(int token, int batch, int rank, int ep) {
  const int chunk = (batch + ep - 1) / ep;
  return token >= rank * chunk && token < (rank + 1) * chunk && token < batch;
}

WEAVE_ONLINE_HD inline float online_slot_sum(const float* slots, int token, int column,
                                             int topk, int hidden) {
  float sum = 0.f;
  for (int slot = 0; slot < topk; ++slot)
    sum += slots[(token * topk + slot) * hidden + column];
  return sum;
}
}  // namespace weave
#undef WEAVE_ONLINE_HD
