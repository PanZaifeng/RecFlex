#pragma once
#include <string>
#include <vector>

namespace onehot {

void OneHotOneBlock(const std::vector<std::string> &inputs, const int tile_size,
                    std::vector<int> &indices, int &num_tiles, int &num_blocks,
                    int &num_inputs);

void OneHotMultiBlock(const std::vector<std::string> &inputs,
                      const int max_concurrent_blocks, const int tile_size,
                      std::vector<int> &indices, int &num_tiles,
                      int &num_blocks, int &num_inputs);

} // namespace onehot