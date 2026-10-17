#pragma once
#include <cstdint>

// Nibble mapping and B128 layout adapted from weave-deepgemm-repro 4fb1e6f:
// deep_gemm/common/mxfp4.cuh and impls/sm90_mxfp4_b_loader_contract.cuh.
// See THIRD_PARTY_NOTICES.md for MIT attribution and upstream provenance.
#ifdef __CUDACC__
#define HOPPER_HD __host__ __device__
#else
#define HOPPER_HD
#endif
namespace weave::hopper {
inline constexpr const char* kNumericContract = "hopper_fp8_e4m3_rne_satfinite_block128_mxfp4_k32_bf16_v1";
HOPPER_HD constexpr uint8_t nibble_to_e4m3(unsigned code) {
  constexpr uint8_t magnitude[] = {0,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c};
  return magnitude[code & 7] | ((code & 8) << 4);
}
HOPPER_HD constexpr uint8_t packed_to_e4m3(uint8_t packed, unsigned k) {
  return nibble_to_e4m3((packed >> (4 * (k & 1))) & 15);
}
HOPPER_HD constexpr unsigned b128_offset(unsigned row, unsigned k) {
  const unsigned offset = row * 128 + k;
  return offset ^ ((offset & 0x380u) >> 3);
}
HOPPER_HD constexpr float activation_scale(float maxabs) {
  return maxabs == 0.f ? 1.f : maxabs / 448.f;
}
HOPPER_HD constexpr bool valid_dimensions(int rank, int ep, int batch, int h,
                                          int i, int experts, int topk) {
  return ep >= 1 && ep <= 4 && rank >= 0 && rank < ep &&
      batch >= 1 && batch <= 32 && h >= 128 && h <= 8192 && h % 128 == 0 &&
      i >= 128 && i <= 8192 && i % 128 == 0 && experts >= topk &&
      experts <= 4096 && experts % ep == 0 && topk >= 1 && topk <= 16;
}
HOPPER_HD constexpr int active_capacity(int batch, int topk, int local_experts) {
  return batch * topk < local_experts ? batch * topk : local_experts;
}
// Called by a single GPU thread. Validate original-width IDs before narrowing.
// Slot order is token-major/top-k-major; no atomics can scramble row order.
// -1 is an invalid routing contract (range or duplicate expert in one token).
template<class Id>
HOPPER_HD inline int plan_routes(const Id* ids, int batch, int topk, int experts,
                                int rank, int ep, int capacity_rows, int* counts,
                                int* active, int* slots) {
  const int local = experts / ep;
  for (int e = 0; e < local; ++e) {
    counts[e] = 0; active[e] = -1;
    for (int row = 0; row < capacity_rows; ++row) slots[e*capacity_rows+row] = -1;
  }
  for (int token = 0; token < batch; ++token) {
    for (int k = 0; k < topk; ++k) {
      const auto id = ids[token*topk+k];
      if (id < 0 || id >= experts) return -1;
      for (int prev = 0; prev < k; ++prev)
        if (ids[token*topk+prev] == id) return -1;
    }
  }
  for (int slot = 0; slot < batch*topk; ++slot) {
    const int id = static_cast<int>(ids[slot]);
    if (id / local != rank) continue;
    const int e = id % local, row = counts[e];
    if (row >= capacity_rows) return -1;
    slots[e*capacity_rows+row] = slot; counts[e] = row+1;
  }
  int total = 0;
  for (int e = 0; e < local; ++e) if (counts[e]) active[total++] = e;
  return total;
}
}  // namespace weave::hopper
#undef HOPPER_HD
