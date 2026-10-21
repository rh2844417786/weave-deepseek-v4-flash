// Weave-inspired active CTA budgeting for native Marlin. Apache-2.0.
// All arithmetic below is independent of CUDA for host/device parity tests.
#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define WEAVE_BUDGET_HD __host__ __device__
#else
#define WEAVE_BUDGET_HD
#endif

struct WeaveBudgetChoice { int workers; int bucket; };

WEAVE_BUDGET_HD inline WeaveBudgetChoice weave_budget_choose(
    int grid, int local_routes, int total_routes, int local_blocks,
    int n_tiles, int k_tiles, int target_percent, int ep_size, int mode) {
  if (local_routes == 0 || local_blocks == 0) return {0, 0};
  if (mode == 2) return {grid, 8};
  const int64_t num = int64_t(grid) * local_routes * ep_size * target_percent;
  const int64_t den = int64_t(total_routes) * 100;
  int desired = int((num + den - 1) / den);
  if (desired > grid) desired = grid;
  if (desired < (grid + 3) / 4) desired = (grid + 3) / 4;
  int bucket = 1;
  while (bucket < 8 && (grid * bucket + 7) / 8 < desired) ++bucket;
  int workers = (grid * bucket + 7) / 8;
  const int64_t work = int64_t(local_blocks) * n_tiles * k_tiles;
  if (work < workers) workers = int(work);
  return {workers, (workers * 8 + grid - 1) / grid};
}

#if defined(__CUDACC__)
// Every full warp computes the same count, so no CTA barrier/shared-memory
// allocation is added. Grouping has already completed on this stream.
// Remote experts are mapped to -1; padding slots contain token IDs >= total.
__device__ inline int weave_budget_local_routes(const int32_t* sorted,
    const int32_t* experts, int padded, int block_size, int total) {
  const int lane = int(threadIdx.x) & 31;
  int count = 0;
  for (int i = lane; i < padded; i += 32) {
    const int token = sorted[i];
    count += (experts[i / block_size] >= 0 && token >= 0 && token < total);
  }
  for (int delta = 16; delta; delta >>= 1)
    count += __shfl_down_sync(0xffffffffu, count, delta);
  return __shfl_sync(0xffffffffu, count, 0);
}
#endif
