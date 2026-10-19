#include "hopper_tile.cuh"
#include <cassert>
#include <cstddef>
#include <vector>

int main() {
  using namespace weave::paper;
  static_assert(TILE_ROWS == 64 && TILE_COLUMNS == 64);
  static_assert(alignof(Shared) == 1024);
  static_assert(offsetof(Shared, a_smem) % 1024 == 0);
  static_assert(offsetof(Shared, b_smem) % 1024 == 0);
  // Check coverage independently of scheduling: each fragment must own one
  // unique valid output, including tails and tiles beyond column zero.
  for (int rows : {1, 17, 32, 63, 64}) {
    for (int first : {0, 64, 192}) {
      std::vector<int> visits(rows * 256, 0);
      for (int thread = 0; thread < 128; ++thread)
        for (int f = 0; f < 32; ++f) {
          const auto p = fragment_element(thread, f, first);
          assert(p.row >= 0 && p.row < 64);
          assert(p.column >= first && p.column < first + 64);
          if (p.row < rows) ++visits[p.row * 256 + p.column];
        }
      for (int row = 0; row < rows; ++row)
        for (int col = 0; col < 256; ++col)
          assert(visits[row * 256 + col] == (col >= first && col < first + 64));
    }
  }
  // Selected PTX accumulator coordinates independently anchor the full cover.
  assert(fragment_element(0, 0, 64).row == 0);
  assert(fragment_element(0, 0, 64).column == 64);
  assert(fragment_element(0, 2, 64).row == 8);
  assert(fragment_element(31, 31, 64).row == 15);
  assert(fragment_element(31, 31, 64).column == 127);
  assert(fragment_element(127, 31, 192).row == 63);
  assert(fragment_element(127, 31, 192).column == 255);
}
