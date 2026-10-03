#pragma once
#ifdef __CUDACC__
#define WEAVE_TILE_HD __host__ __device__
#define WEAVE_TILE_INLINE __device__ __forceinline__
#else
#define WEAVE_TILE_HD
#define WEAVE_TILE_INLINE inline
#endif

namespace weave {
constexpr int TILE_ROWS=16, TILE_COLUMNS=64;
struct TileState { int gate_next,gate_done,down_next,down_done; };
struct TileClaim { int group,stage,column; }; // -1 drained, -2 work inflight.
WEAVE_TILE_HD inline int tile_columns(int width) { return (width+TILE_COLUMNS-1)/TILE_COLUMNS; }
WEAVE_TILE_HD inline int tile_group_capacity(int tasks,int experts,int chunks) {
  return (tasks+TILE_ROWS-1)/TILE_ROWS+experts*chunks;
}

// A claimant never blocks while holding a down job whose input is incomplete.
// Round-robin starts avoid rescanning thousands of already-drained groups.
template<class Load,class Add>
WEAVE_TILE_INLINE TileClaim claim_tile(TileState* states,int count,int gate_tiles,
                                     int down_tiles,Load load,Add add,int* cursor=nullptr) {
  if(count==0) return {-1,0,0};
  int start=cursor ? add(cursor,1)%count : 0;
  bool pending=false;
  for(int q=0;q<count;++q) {
    int group=(start+q)%count;
    TileState& state=states[group];
    if(load(&state.gate_next)<gate_tiles) {
      int column=add(&state.gate_next,1);
      if(column<gate_tiles) return {group,0,column};
    }
    if(load(&state.gate_done)>=gate_tiles && load(&state.down_next)<down_tiles) {
      int column=add(&state.down_next,1);
      if(column<down_tiles) return {group,1,column};
    }
    pending=pending || load(&state.down_done)<down_tiles;
  }
  return {pending ? -2 : -1,0,0};
}
}
#undef WEAVE_TILE_HD
#undef WEAVE_TILE_INLINE
