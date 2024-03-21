#pragma once
#include <cstdlib>
#include <cub/cub.cuh>
#include <string>

template <typename T>
int get_max_concurrent_blocks(const T &knl, int block_threads,
                              size_t dyn_smem_size = 0,
                              bool print_flag = false) {
  int max_sm_occupancy, num_sm;
  CubDebugExit(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_sm_occupancy, knl, block_threads, dyn_smem_size));
  CubDebugExit(
      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
  int max_concurrent_blocks = max_sm_occupancy * num_sm;
  if (print_flag) {
    printf("max_sm_occupancy = %d\n", max_sm_occupancy);
    printf("max_concurrent_blocks = %d\n", max_concurrent_blocks);
  }
  return max_concurrent_blocks;
}

template <typename T>
size_t get_dyn_smem_size(const T &knl, int occupancy, int block_threads) {
  int max_sm_occupancy;
  CubDebugExit(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_sm_occupancy, knl, block_threads, 0));

  size_t dyn_smem_size;
  if (max_sm_occupancy < occupancy) {
    printf("WARNING: target occupancy %d cannot be achieved (max %d)\n",
           occupancy, max_sm_occupancy);
    dyn_smem_size = 0;
  } else if (max_sm_occupancy == occupancy) {
    dyn_smem_size = 0;
  } else {
    // least value
    CubDebugExit(cudaOccupancyAvailableDynamicSMemPerBlock(
        &dyn_smem_size, knl, occupancy + 1, block_threads));
    dyn_smem_size += 1;

    cudaFuncAttributes attr;
    CubDebugExit(cudaFuncGetAttributes(&attr, knl));
    while (dyn_smem_size > attr.maxDynamicSharedSizeBytes) {
      CubDebugExit(cudaFuncSetAttribute(
          knl, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_size));
      CubDebugExit(cudaFuncGetAttributes(&attr, knl));
      int real_occupancy;
      CubDebugExit(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &real_occupancy, knl, block_threads, dyn_smem_size));
      if (real_occupancy > occupancy) {
        dyn_smem_size += 1024;
      }
    }
  }

  return dyn_smem_size;
}

__device__ __host__ __forceinline__ static size_t alignmem(size_t x) {
  return (x + 127) / 128 * 128;
}

template <typename T> __device__ __host__ size_t align_need_pad_elem(size_t n) {
  constexpr size_t align_base = 128 / sizeof(T);
  static_assert(align_base * sizeof(T) == 128, "Invalid type");
  return ((n + align_base - 1) / align_base * align_base) - n;
}

std::string get_env(const std::string &key, const std::string &default_val);

bool exist_file(const std::string &fname);