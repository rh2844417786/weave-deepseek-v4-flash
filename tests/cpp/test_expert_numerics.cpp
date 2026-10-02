#include "expert_numerics.h"
#include <cstdint>
#include <cstring>
#include <iostream>

// Host-only BF16 round-to-nearest-even, independent of CUDA intrinsics.
struct RoundBF16 {
  float operator()(float value) const {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
};

int main() {
  int failed = 0;
  auto check = [&](float actual, float expected, const char* boundary) {
    if (actual != expected) {
      std::cerr << boundary << ": " << actual << " != " << expected << '\n';
      ++failed;
    }
  };
  check(weave::swiglu_product(.0625f, .6875f, 10.f, RoundBF16{}),
        .022216796875f, "BF16 SiLU before up multiply");
  check(weave::swiglu_product(.0625f, 1.875f, 10.f, RoundBF16{}),
        .060546875f, "BF16 SiLU second witness");
  check(weave::routed_product(1.0078125f, .1f, RoundBF16{}),
        .10107421875f, "BF16 routing weight and product");
  check(weave::routed_product(1.f, .3333f, RoundBF16{}),
        .333984375f, "routing weight rounds before multiply");
  check(weave::routed_product(1.0078125f, 0.f, RoundBF16{}), 0.f, "zero route");
  check(weave::swiglu_product(12.f, -12.f, 10.f, RoundBF16{}), -100.f, "clamp");
  return failed ? 1 : 0;
}
