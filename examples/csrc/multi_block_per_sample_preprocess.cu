#include "multi_block_per_sample_preprocess.cuh"
#include <cub/cub.cuh>

namespace mbs {

void StringSplitToInt(const std::vector<std::string> &inputs,
                      const int max_concurrent_blocks, const int tile_size,
                      std::vector<int> &indices,
                      std::vector<int> &segment_offsets,
                      std::vector<int> &workload_mapping, int &num_tiles,
                      int &num_blocks, int &num_inputs, int &num_segments) {
  const int bs = inputs.size();
  indices = std::vector<int>();
  segment_offsets = std::vector<int>(bs + 1, 0);
  workload_mapping = std::vector<int>();

  num_tiles = 0;
  for (int seg_id = 0; seg_id < bs; ++seg_id) {
    int index = 0;
    int count = 0;
    for (char ch : inputs[seg_id]) {
      if (ch < '0' || ch > '9') {
        indices.push_back(index);
        ++count;
        index = 0;
      } else {
        index = index * 10 + (ch - '0');
      }
    }
    if (inputs[seg_id].size() > 0) {
      indices.push_back(index);
      ++count;
    }

    segment_offsets[seg_id + 1] = segment_offsets[seg_id] + count;
    int tiles_for_sample = cub::DivideAndRoundUp(count, tile_size);
    num_tiles += tiles_for_sample;
    for (int sample_tile_idx = 0; sample_tile_idx < tiles_for_sample;
         ++sample_tile_idx) {
      workload_mapping.push_back(seg_id);
      workload_mapping.push_back(sample_tile_idx);
    }
  }

  num_inputs = indices.size();
  num_segments = bs;
  num_blocks = std::min(num_tiles, max_concurrent_blocks);
}

} // namespace mbs