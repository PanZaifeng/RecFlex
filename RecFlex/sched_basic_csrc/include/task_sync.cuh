#pragma once
#include <cub/cub.cuh>

template <int BLOCK_THREADS>
__device__ __forceinline__ void
task_sync(int *__restrict__ d_sync, const int tile_idx, const int total_tiles) {
  volatile int *d_vol_sync = d_sync;

  // Threadfence and syncthreads to make sure global writes are visible before
  // thread-0 reports in with its sync counter
  __threadfence();
  __syncthreads();

  if (tile_idx == 0) {
    // Report in ourselves
    if (threadIdx.x == 0) {
      d_sync[tile_idx] = 1;
    }

    __syncthreads();

    // Wait for everyone else to report in
    for (int peer_block = threadIdx.x; peer_block < total_tiles;
         peer_block += BLOCK_THREADS) {
      while (cub::ThreadLoad<cub::LOAD_CG>(d_sync + peer_block) == 0) {
        __threadfence_block();
      }
    }

    __syncthreads();

    // Let everyone know it's safe to proceed
    for (int peer_block = threadIdx.x; peer_block < total_tiles;
         peer_block += BLOCK_THREADS) {
      d_vol_sync[peer_block] = 0;
    }
  } else {
    if (threadIdx.x == 0) {
      // Report in
      d_vol_sync[tile_idx] = 1;

      // Wait for acknowledgment
      while (cub::ThreadLoad<cub::LOAD_CG>(d_sync + tile_idx) == 1) {
        __threadfence_block();
      }
    }

    __syncthreads();
  }
}

template <int EMBED_DIM, int BLOCK_THREADS, typename Tparam>
__device__ __forceinline__ void InitOutputs(Tparam *__restrict__ d_outputs,
                                            const int num_segments,
                                            int task_bid, int task_blocks) {
  for (int idx = (task_bid * BLOCK_THREADS) + threadIdx.x;
       idx < num_segments * EMBED_DIM; idx += task_blocks * BLOCK_THREADS) {
    d_outputs[idx] = 0;
  }
}