#include "onehot_preprocess.cuh"
#include <cub/cub.cuh>

namespace onehot {

void OneHotOneBlock(const std::vector<std::string> &inputs, const int tile_size,
                    std::vector<int> &indices, int &num_tiles, int &num_blocks,
                    int &num_inputs) {
  indices = std::vector<int>(inputs.size());
  for (int i = 0; i < inputs.size(); ++i)
    indices[i] = std::stoi(inputs[i]);
  num_inputs = indices.size();
  num_tiles = cub::DivideAndRoundUp(num_inputs, tile_size);
  num_blocks = 1;
}

void OneHotMultiBlock(const std::vector<std::string> &inputs,
                      const int max_concurrent_blocks, const int tile_size,
                      std::vector<int> &indices, int &num_tiles,
                      int &num_blocks, int &num_inputs) {
  indices = std::vector<int>(inputs.size());
  for (int i = 0; i < inputs.size(); ++i)
    indices[i] = std::stoi(inputs[i]);
  num_inputs = indices.size();
  num_tiles = cub::DivideAndRoundUp(num_inputs, tile_size);
  num_blocks = std::min(num_tiles, max_concurrent_blocks);
}

} // namespace onehot