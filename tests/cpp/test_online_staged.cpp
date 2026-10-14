#include <cassert>
#include <iostream>
#include <vector>
#include "online_staged.h"

int main(int argc, char**) {
  if (argc > 1) {
    int b, k, rank, ep, experts;
    std::cin >> b >> k >> rank >> ep >> experts;
    std::vector<int> ids(b * k);
    for (int& id : ids) std::cin >> id;
    const int capacity = weave::online_group_capacity(b, k, experts / ep);
    std::vector<int> meta(capacity * 2), slots(capacity * weave::TILE_ROWS);
    int count = weave::online_build_groups(ids.data(), b, k, rank, ep, experts,
                                          capacity, meta.data(), slots.data());
    std::cout << capacity << ' ' << count;
    for (int value : meta) std::cout << ' ' << value;
    for (int value : slots) std::cout << ' ' << value;
    return 0;
  }
  // A full expert must acquire a second group after all first encounters.
  std::vector<int> ids(34);
  for (int j = 0; j < 17; ++j) { ids[2*j] = 2; ids[2*j+1] = 0; }
  const int capacity = weave::online_group_capacity(17, 2, 4);
  std::vector<int> meta(capacity * 2), slots(capacity * 16);
  assert(weave::online_build_groups(ids.data(), 17, 2, 0, 2, 8,
                                    capacity, meta.data(), slots.data()) == 4);
  assert(meta[0] == 2 && meta[1] == 16 && meta[2] == 0 && meta[3] == 16);
  assert(meta[4] == 2 && meta[5] == 1 && meta[6] == 0 && meta[7] == 1);
  assert(slots[32] == 32 && slots[48] == 33 && slots[33] == -1);
  // Underallocation must fail before a write reaches the canary.
  int tiny_meta[] = {77, 77, 777}, tiny_slots[17];
  for (int& value : tiny_slots) value = 777;
  assert(weave::online_build_groups(ids.data(), 17, 2, 0, 2, 8,
                                    1, tiny_meta, tiny_slots) == -3);
  assert(tiny_meta[2] == 777 && tiny_slots[16] == 777);
  // Invalid remote routes and duplicate slots are rejected on every rank.
  const int invalid[] = {7, 8}, duplicate[] = {7, 7};
  assert(weave::online_build_groups(invalid, 1, 2, 0, 2, 8, 1,
                                    tiny_meta, tiny_slots) == -2);
  assert(weave::online_build_groups(duplicate, 1, 2, 0, 2, 8, 1,
                                    tiny_meta, tiny_slots) == -2);
  // Int64 source IDs must be checked before conversion can wrap into range.
  const long long wide_invalid[] = {4294967303LL, 2}, wide_valid[] = {7, 2};
  int narrowed[] = {77, 77};
  assert(!weave::online_copy_ids(wide_invalid, narrowed, 2, 8));
  assert(weave::online_copy_ids(wide_valid, narrowed, 2, 8));
  assert(narrowed[0] == 7 && narrowed[1] == 2);
  // Origin shards are contiguous ceil(B/EP), including empty small-B ranks.
  for (int b : {1, 7, 17, 32}) for (int token = 0; token < b; ++token) {
    int owners = 0;
    for (int rank = 0; rank < 4; ++rank)
      owners += weave::online_origin_row(token, b, rank, 4);
    assert(owners == 1);
    assert(weave::online_origin_row(token, b, token / ((b + 3) / 4), 4));
  }
  // Order is observable in FP32: reassociation drops the final one.
  const float products[] = {16777216.f, -16777216.f, 1.f};
  assert(weave::online_slot_sum(products, 0, 0, 3, 1) == 1.f);
}
