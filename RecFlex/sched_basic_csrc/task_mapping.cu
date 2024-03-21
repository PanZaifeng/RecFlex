#include <vector>
#include <numeric>

#include "task_mapping.cuh"

void GetTaskMapping(const std::vector<int> &task_blocks,
                    std::vector<int2> &task_mapping) {
  int total_blocks = std::accumulate(task_blocks.begin(), task_blocks.end(), 0);
  task_mapping = std::vector<int2>(total_blocks);

  int offset = 0;
  for (int feat_idx = 0; feat_idx < task_blocks.size(); ++feat_idx) {
    for (int i = 0; i < task_blocks[feat_idx]; ++i) {
      task_mapping[offset + i] = make_int2(feat_idx, i);
    }
    offset += task_blocks[feat_idx];
  }
}