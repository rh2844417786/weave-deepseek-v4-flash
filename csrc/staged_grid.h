#pragma once
#include "tile_scheduler.h"

#ifdef __CUDACC__
#define WEAVE_STAGED_HD __host__ __device__
#else
#define WEAVE_STAGED_HD
#endif

namespace weave {
struct StagedTile { int group, column; };
WEAVE_STAGED_HD inline int staged_tiles(int groups, int columns) {
  return groups * columns;
}
WEAVE_STAGED_HD inline int staged_blocks(int grid, int capacity, int columns) {
  const int direct = capacity * columns;
  return grid ? grid : (direct > 0 ? direct : 1);
}
WEAVE_STAGED_HD inline StagedTile staged_tile(int index, int columns) {
  return {index / columns, index % columns};
}
}  // namespace weave
#undef WEAVE_STAGED_HD
