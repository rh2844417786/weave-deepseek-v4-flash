#pragma once
// Resident CTA Hopper expert tiles. Arithmetic is adapted from
// experimental/hopper_fp8/kernel.cu and weave-deepgemm-repro 4fb1e6f
// (MIT; see THIRD_PARTY_NOTICES.md). CUTLASS comes from the pinned vendor tree.
// No launch or allocation occurs here. A is gathered and quantized again for
// every output tile; this cost is not covered by standalone Hopper timings.
#include <cstddef>
#include <cstdint>
#include "../hopper_fp8/math_contract.h"

namespace weave::paper {
inline constexpr int TILE_ROWS = 64;
inline constexpr int TILE_COLUMNS = 64;

// The portable coordinate helper is also used by the actual device stores.
struct Element { int row, column; };
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr Element fragment_element(int thread, int fragment, int first) {
  const int lane = thread % 32, warp = thread / 32;
  return {warp * 16 + lane / 4 + ((fragment % 4) / 2) * 8,
          first + (fragment / 4) * 8 + (lane % 4) * 2 + fragment % 2};
}

// All WGMMA descriptor bases must have base_offset=0 for the B128 swizzle.
// The owning persistent kernel provides dynamic shared memory with this
// alignment and may reserve more bytes to enforce one resident CTA per SM.
struct alignas(1024) Shared {
  alignas(1024) uint8_t a_smem[64 * 128];
  alignas(1024) uint8_t b_smem[64 * 128];
  alignas(128) uint8_t b_tma[64 * 64];
  float a_scale[64];
  float warp_max[4];
  alignas(16) uint64_t barrier_storage;
};
static_assert(alignof(Shared) == 1024);
static_assert(offsetof(Shared, a_smem) % 1024 == 0);
static_assert(offsetof(Shared, b_smem) % 1024 == 0);
}  // namespace weave::paper

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include "../../csrc/expert_numerics.h"

namespace weave::paper {
// Host-only encoding during weight binding. The caller keeps descriptors in
// stable device/global or __grid_constant__ parameter memory throughout use;
// a pointer into an ordinary host object is not a valid TMA descriptor.
inline CUtensorMap tensor_map(void* address, int width, int rows, int experts,
                              int box_width) {
  if (!address || reinterpret_cast<uintptr_t>(address) % 16 || width <= 0 ||
      width % 16 || rows <= 0 || experts <= 0 || box_width <= 0 ||
      box_width % 16 || box_width > 256)
    throw std::invalid_argument("Hopper persistent TMA: invalid aligned byte tensor shape");
  CUtensorMap map{};
  const cuuint64_t dims[3] = {cuuint64_t(width), cuuint64_t(rows), cuuint64_t(experts)};
  const cuuint64_t strides[2] = {cuuint64_t(width), cuuint64_t(width) * rows};
  const cuuint32_t box[3] = {cuuint32_t(box_width), 64, 1};
  const cuuint32_t element_strides[3] = {1, 1, 1};
  const CUresult status = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3, address, dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string("Hopper persistent tensor map: ") +
                             (message ? message : "driver failure"));
  }
  return map;
}

namespace tile_detail {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct RoundBF16 {
  __device__ __forceinline__ float operator()(float value) const {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
};
__device__ __forceinline__ void require_finite(float value) {
  if (!isfinite(value)) {
    assert(false && "Hopper persistent tile produced or received nonfinite data");
    asm volatile("trap;");
  }
}
__device__ __forceinline__ uint64_t descriptor(uint8_t* pointer) {
  cute::GmmaDescriptor desc{};
  desc.bitfield.start_address_ = static_cast<uint32_t>(__cvta_generic_to_shared(pointer)) >> 4;
  desc.bitfield.leading_byte_offset_ = 0;
  desc.bitfield.stride_byte_offset_ = 1024 >> 4;
  desc.bitfield.base_offset_ = 0;
  desc.bitfield.layout_type_ = 1;
  return desc;
}
template<size_t... I>
__device__ __forceinline__ void mma(uint64_t a, uint64_t b, float* accum,
                                   std::index_sequence<I...>) {
  cute::SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>::fma(
      a, b, accum[I]..., cute::SM90::GMMA::ScaleOut::Zero);
}
__device__ __forceinline__ float e8m0(uint8_t code) {
  if (code == 255) {
    assert(false && "Hopper MXFP4 E8M0 NaN scale is unsupported");
    asm volatile("trap;");
  }
  return ldexpf(1.f, int(code) - 127);
}

// Every row uses all 128 threads for the same maxabs/448 quantization as
// the original candidate. Padding never calls the gather functor.
template<class Input>
__device__ __forceinline__ void quantize(int rows, int stage, Input input, Shared& s) {
  const int tid = threadIdx.x;
  for (int row = 0; row < rows; ++row) {
    const float value = __bfloat162float(input(row, stage * 128 + tid));
    require_finite(value);
    float maximum = fabsf(value);
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    if ((tid & 31) == 0) s.warp_max[tid / 32] = maximum;
    __syncthreads();
    maximum = fmaxf(fmaxf(s.warp_max[0], s.warp_max[1]),
                    fmaxf(s.warp_max[2], s.warp_max[3]));
    const float scale = weave::hopper::activation_scale(maximum);
    if (tid == 0) s.a_scale[row] = scale;
    s.a_smem[weave::hopper::b128_offset(row, tid)] =
        __nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3);
    __syncthreads();
  }
  for (int row = rows; row < TILE_ROWS; ++row) {
    s.a_smem[weave::hopper::b128_offset(row, tid)] = 0;
    if (tid == 0) s.a_scale[row] = 0.f;
  }
  __syncthreads();
}

__device__ __forceinline__ void init(Shared& s) {
  if (threadIdx.x == 0) {
    reinterpret_cast<Barrier*>(&s.barrier_storage)->init(1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
}
__device__ __forceinline__ void finish(Shared& s) {
  // All TMA phases and WGMMA groups were waited before reaching this point.
  // Explicit invalidation permits this memory to be reinitialized next tile.
  __syncthreads();
  if (threadIdx.x == 0) Barrier::invalidate(&s.barrier_storage);
  __syncthreads();
}

// One packed-weight K128 transaction, then four ordered K32 contributions.
// phase is CTA-uniform and remains live across gate/up transactions.
__device__ __forceinline__ void accumulate(const CUtensorMap* map,
    const uint8_t* scales, int expert, int n, int k, int first, int stage,
    float* result, Shared& s, unsigned& phase) {
  auto* barrier = reinterpret_cast<Barrier*>(&s.barrier_storage);
  constexpr uint64_t hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
  if (threadIdx.x == 0) {
    barrier->arrive_and_expect_tx(64 * 64);
    cute::SM90_TMA_LOAD_3D::copy(map, &s.barrier_storage, hint, s.b_tma,
                                stage * 64, first, expert);
  }
  barrier->wait(phase);
  __syncthreads();
  for (int index = threadIdx.x; index < 64 * 128; index += 128) {
    const int row = index / 128, col = index % 128;
    s.b_smem[weave::hopper::b128_offset(row, col)] =
        weave::hopper::packed_to_e4m3(s.b_tma[row * 64 + col / 2], col);
  }
  __syncthreads();
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    float partial[32] = {};
    #pragma unroll
    for (int z = 0; z < 32; ++z) asm volatile("" : "+f"(partial[z]) :: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    mma(descriptor(s.a_smem + group * 32), descriptor(s.b_smem + group * 32),
        partial, std::make_index_sequence<32>{});
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for (int z = 0; z < 32; ++z) asm volatile("" : "+f"(partial[z]) :: "memory");
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
      const Element p = fragment_element(threadIdx.x, z, first);
      const float as = s.a_scale[p.row];
      const float bs = e8m0(scales[(size_t(expert) * n + p.column) * (k / 32) +
                                   stage * 4 + group]);
      result[z] = __fadd_rn(result[z], __fmul_rn(__fmul_rn(partial[z], as), bs));
    }
  }
  // Release B and A before another TMA/quantization can overwrite them.
  __syncthreads();
  phase ^= 1;
}
__device__ __forceinline__ void check_shape(int rows, int h, int i, int first, int n) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      rows < 0 || rows > 64 || h <= 0 || h % 128 || i <= 0 || i % 128 ||
      first < 0 || first % 64 || first + 64 > n) {
    assert(false && "Hopper persistent tile requires CTA128 M64N64 and H/I multiples128");
    asm volatile("trap;");
  }
}
}  // namespace tile_detail

// Each call is CTA-uniform. Input and Store use group-local rows and absolute
// matrix columns; first is the output tile base, never a per-warp offset.
template<class Input, class Store>
__device__ void gate_up(const CUtensorMap* gate_map, const CUtensorMap* up_map,
    const uint8_t* s1, const uint8_t* s3, int expert, int rows, int h, int i,
    int first, float clamp, Input input, Store store, Shared& scratch,
    bool arithmetic = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  tile_detail::check_shape(rows, h, i, first, i);
  if (!arithmetic) {
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
      const Element p = fragment_element(threadIdx.x, z, first);
      if (p.row < rows) store(p.row, p.column, __float2bfloat16_rn(0.f));
    }
    __syncthreads();
    return;
  }
  tile_detail::init(scratch);
  float gate[32] = {}, up[32] = {};
  unsigned phase = 0;
  for (int stage = 0; stage < h / 128; ++stage) {
    tile_detail::quantize(rows, stage, input, scratch);
    tile_detail::accumulate(gate_map, s1, expert, i, h, first, stage, gate, scratch, phase);
    tile_detail::accumulate(up_map, s3, expert, i, h, first, stage, up, scratch, phase);
  }
  #pragma unroll
  for (int z = 0; z < 32; ++z) {
    const Element p = fragment_element(threadIdx.x, z, first);
    if (p.row < rows) {
      const float g = tile_detail::RoundBF16{}(gate[z]);
      const float u = tile_detail::RoundBF16{}(up[z]);
      tile_detail::require_finite(g);
      tile_detail::require_finite(u);
      const float value = weave::swiglu_product(g, u, clamp, tile_detail::RoundBF16{});
      tile_detail::require_finite(value);
      store(p.row, p.column, __float2bfloat16_rn(value));
    }
  }
  tile_detail::finish(scratch);
#else
  asm volatile("trap;");
#endif
}

template<class Input, class Store>
__device__ void down(const CUtensorMap* down_map, const uint8_t* s2, int expert,
    int rows, int h, int i, int first, Input input, Store store, Shared& scratch,
    bool arithmetic = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  tile_detail::check_shape(rows, h, i, first, h);
  if (!arithmetic) {
    #pragma unroll
    for (int z = 0; z < 32; ++z) {
      const Element p = fragment_element(threadIdx.x, z, first);
      if (p.row < rows) store(p.row, p.column, 0.f);
    }
    __syncthreads();
    return;
  }
  tile_detail::init(scratch);
  float result[32] = {};
  unsigned phase = 0;
  for (int stage = 0; stage < i / 128; ++stage) {
    tile_detail::quantize(rows, stage, input, scratch);
    tile_detail::accumulate(down_map, s2, expert, h, i, first, stage, result, scratch, phase);
  }
  #pragma unroll
  for (int z = 0; z < 32; ++z) {
    const Element p = fragment_element(threadIdx.x, z, first);
    if (p.row < rows) {
      const float value = tile_detail::RoundBF16{}(result[z]);
      tile_detail::require_finite(value);
      // Routing is applied exactly once by the origin combine stage.
      store(p.row, p.column, value);
    }
  }
  tile_detail::finish(scratch);
#else
  asm volatile("trap;");
#endif
}
}  // namespace weave::paper
#endif  // __CUDACC__
