#pragma once
#include <string>
#include <vector>

namespace mbs {

void StringSplitToInt(const std::vector<std::string> &inputs,
                      const int max_concurrent_blocks, const int tile_size,
                      std::vector<int> &indices,
                      std::vector<int> &segment_offsets,
                      std::vector<int> &workload_mapping, int &num_tiles,
                      int &num_blocks, int &num_inputs, int &num_segments);

void StringSplitToIntOneBlock(const std::vector<std::string> &inputs,
                              std::vector<int> &indices,
                              std::vector<int> &segment_offsets, int &num_tiles,
                              int &num_blocks, int &num_inputs,
                              int &num_segments);

} // namespace mbs