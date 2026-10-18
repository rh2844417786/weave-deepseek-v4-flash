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
namespace weave::hopper_prefill {
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
      batch >= 1 && batch <= 8192 && h >= 128 && h <= 8192 && h % 128 == 0 &&
      i >= 128 && i <= 8192 && i % 128 == 0 && experts >= topk &&
      experts <= 4096 && experts % ep == 0 && topk >= 1 && topk <= 16;
}
HOPPER_HD constexpr int active_capacity(int batch, int topk, int local_experts) {
  return batch * topk < local_experts ? batch * topk : local_experts;
}
HOPPER_HD constexpr int tile_capacity(int batch, int topk, int local) {
  return (batch*topk+63)/64+local;
}
HOPPER_HD constexpr int tile_launch_capacity(int batch, int topk, int local) {
  return tile_capacity(batch,topk,active_capacity(batch,topk,local));
}
HOPPER_HD constexpr int token_range_start(int batch, int thread) {
  return batch*thread/256;
}
} // namespace weave::hopper_prefill
#undef HOPPER_HD
