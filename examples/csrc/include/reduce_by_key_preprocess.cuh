#pragma once
#include <cub/cub.cuh>
#include <numeric>
#include <string>
#include <vector>

#include "reduce_by_key_kernel.cuh"

namespace rbk {

template <int SCAN_DIM, typename Tparam>
void StringSplitToInt(const std::vector<std::string> &inputs,
                      const int max_concurrent_blocks, const int tile_size,
                      std::vector<int> &indices, std::vector<int> &segment_ids,
                      int &num_tiles, int &num_blocks, int &num_inputs,
                      int &num_segments, int &tile_status_storage_bytes) {
  const int bs = inputs.size();
  indices = std::vector<int>();
  segment_ids = std::vector<int>();

  for (int seg_id = 0; seg_id < bs; ++seg_id) {
    int index = 0;
    for (char ch : inputs[seg_id]) {
      if (ch < '0' || ch > '9') {
        indices.push_back(index);
        segment_ids.push_back(seg_id);
        index = 0;
      } else {
        index = index * 10 + (ch - '0');
      }
    }
    if (inputs[seg_id].size() > 0) {
      indices.push_back(index);
      segment_ids.push_back(seg_id);
    }
  }

  num_inputs = indices.size();
  num_segments = bs;
  num_tiles = cub::DivideAndRoundUp(num_inputs, tile_size);
  num_blocks = std::min(num_tiles, max_concurrent_blocks);
  tile_status_storage_bytes =
      rbk::GetAllocationSize<rbk::ScanVecPair<SCAN_DIM, Tparam>>(num_tiles);
}

} // namespace rbk