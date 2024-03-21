#pragma once
#include <string>
#include <vector>

namespace obms {

void StringSplitToInt(const std::vector<std::string> &inputs, int tile_size,
                      std::vector<int> &indices,
                      std::vector<int> &segment_offsets, int &num_tiles,
                      int &num_blocks, int &num_inputs, int &num_segments);

} // namespace obms