#pragma once
#include <cmath>

#ifdef __CUDACC__
#define WEAVE_NUMERIC_INLINE __device__ __forceinline__
#else
#define WEAVE_NUMERIC_INLINE inline
#endif

namespace weave {
// Inputs gate, up and down are already rounded to BF16 by GEMM storage.
template <class Round>
WEAVE_NUMERIC_INLINE float swiglu_product(float gate, float up, float limit, Round round) {
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  // Native swiglu_limit_func materializes F.silu(BF16 gate) before the product.
  float activation = round(gate / (1.f + expf(-gate)));
  return round(activation * up);
}

template <class Round>
WEAVE_NUMERIC_INLINE float routed_product(float down, float weight, Round round) {
  // Marlin converts routing to the activation dtype, then uses BF16 __hmul2.
  return round(down * round(weight));
}
}  // namespace weave
#undef WEAVE_NUMERIC_INLINE
