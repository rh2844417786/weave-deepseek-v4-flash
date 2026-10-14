#include <cassert>
#include <vector>
#include "staged_grid.h"

int main() {
  // Zero local work, one partial output tile, full production widths, and
  // more tasks than the largest fixed grid. Capacity exceeds live count.
  for (int groups : {0, 1, 17, 192}) {
    for (int width : {32, 64, 96, 2048, 4096}) {
      const int columns = weave::tile_columns(width);
      for (int grid : {0, 132, 264, 528}) {
        const int capacity = groups + 3;
        const int blocks = weave::staged_blocks(grid, capacity, columns);
        assert(blocks == (grid == 0 ? capacity * columns : grid));
        std::vector<int> visits(groups * columns, 0);
        for (int block = 0; block < blocks; ++block) {
          for (int tile = block; tile < weave::staged_tiles(groups, columns); tile += blocks) {
            const auto claim = weave::staged_tile(tile, columns);
            assert(claim.group >= 0 && claim.group < groups);
            assert(claim.column >= 0 && claim.column < columns);
            ++visits[claim.group * columns + claim.column];
          }
        }
        for (int count : visits) assert(count == 1);
      }
    }
  }
  assert(weave::staged_blocks(0, 0, 1) == 1);
}
