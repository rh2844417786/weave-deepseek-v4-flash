#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <cstddef>
#include "expert_numerics.h"

namespace weave {
namespace tile_math {

struct RoundBF16 {
  __device__ __forceinline__ float operator()(float value) const {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
};

struct Weights {
  const unsigned char *w1, *w3, *w2, *s1, *s3, *s2;
};

// Original MXFP4 nibble and E8M0 decoding; no LUT or fast-math substitution.
__device__ inline float decode4(unsigned char byte, int nibble, unsigned char scale) {
  int code = (byte >> (4 * nibble)) & 15;
  constexpr float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = values[code & 7];
  if (code & 8) v = -v;
  return scale == 255 ? CUDART_NAN_F : ldexpf(v, int(scale) - 127);
}

__device__ inline float weight(const unsigned char* w, const unsigned char* s,
                              size_t row, int col, int width) {
  return decode4(w[row * (width / 2) + col / 2], col & 1,
                 s[row * (width / 32) + col / 32]);
}

// A warp computes M16 x N16. Four independent warps form M16 x N64.
// Input(row, column) supplies BF16 values; scratch is 1024 BF16 per warp:
// A[256], B[256], and FP32 accumulator storage[256]. This is the original
// load / mma / store order, including padding and BF16 weight conversion.
template<class Input>
__device__ inline void group_mma(const unsigned char* weights, const unsigned char* scales,
                                int expert, int rows, int width, int outputs,
                                int first_column, Input input, __nv_bfloat16* scratch) {
  using namespace nvcuda;
  int lane = threadIdx.x % 32;
  auto a = scratch;
  auto b = a + 256;
  auto result = reinterpret_cast<float*>(b + 256);
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
  wmma::fill_fragment(accum, 0.f);
  for (int column = 0; column < width; column += 16) {
    for (int j = lane; j < 256; j += 32) {
      int row = j / 16, k = j % 16;
      __nv_bfloat16 value = __float2bfloat16_rn(0.f);
      if (row < rows) value = input(row, column + k);
      a[j] = value;
      int output = first_column + j / 16;
      b[j] = output < outputs ? __float2bfloat16_rn(weight(weights, scales,
          size_t(expert) * outputs + output, column + j % 16, width)) : __float2bfloat16_rn(0.f);
    }
    __syncwarp();
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(accum, af, bf, accum);
    __syncwarp();
  }
  wmma::store_matrix_sync(result, accum, 16, wmma::mem_row_major);
  __syncwarp();
}

template<class Input, class Store>
__device__ inline void gate_up(Weights weights, int expert, int rows, int h, int i,
                              int first, float clamp, Input input, Store store,
                              __nv_bfloat16* scratch, bool compute = true) {
  int lane = threadIdx.x % 32;
  auto result = reinterpret_cast<float*>(scratch + 512);
  float gate[8];
  if (compute) {
    group_mma(weights.w1, weights.s1, expert, rows, h, i, first, input, scratch);
    for (int q = 0; q < 8; ++q) gate[q] = result[lane + 32 * q];
    group_mma(weights.w3, weights.s3, expert, rows, h, i, first, input, scratch);
  }
  for (int q = 0; q < 8; ++q) {
    int j = lane + 32 * q, row = j / 16, column = first + j % 16;
    if (row < rows && column < i) {
      float value = 0.f;
      if (compute) {
        float g = RoundBF16{}(gate[q]), u = RoundBF16{}(result[j]);
        value = weave::swiglu_product(g, u, clamp, RoundBF16{});
      }
      store(row, column, __float2bfloat16_rn(value));
    }
  }
}

template<class Input, class Store>
__device__ inline void down(Weights weights, int expert, int rows, int h, int i,
                           int first, Input input, Store store,
                           __nv_bfloat16* scratch, bool compute = true) {
  int lane = threadIdx.x % 32;
  auto result = reinterpret_cast<float*>(scratch + 512);
  if (compute) group_mma(weights.w2, weights.s2, expert, rows, i, h, first, input, scratch);
  for (int j = lane; j < 256; j += 32) {
    int row = j / 16, column = first + j % 16;
    if (row < rows && column < h)
      store(row, column, compute ? RoundBF16{}(result[j]) : 0.f);
  }
}

}  // namespace tile_math
}  // namespace weave
