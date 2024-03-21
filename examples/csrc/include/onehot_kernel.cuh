#pragma once
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>

#include "task_sync.cuh"

namespace onehot {

template <int BLOCK_THREADS> struct GatherSegmentsTempStorage {
  int indices[BLOCK_THREADS];
};

template <bool FULL_TILE, int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT>
__device__ __forceinline__ void
GatherSegmentsCore(GatherSegmentsTempStorage<BLOCK_THREADS> &s,
                   const Tparam *__restrict__ d_params,
                   const AccessIndicesT &access_indices, const int num_inputs,
                   const int tile_idx, Tparam *__restrict__ d_outputs,
                   const int embed_dim_offset) {
  constexpr int TILE_SIZE = BLOCK_THREADS;
  const int tile_offset = tile_idx * TILE_SIZE;
  const int tid = threadIdx.x;
  const int num_left_outputs = (num_inputs - tile_offset) * EMBED_DIM;

  const int input_idx = tile_offset + tid;
  if (FULL_TILE || input_idx < num_inputs) {
    s.indices[tid] = access_indices(input_idx);
  }
  __syncthreads();

  auto *d_tile_outputs = d_outputs + tile_offset * EMBED_DIM_SUM;
#pragma unroll UNROLL_FACTOR
  for (int i = 0; i < EMBED_DIM; ++i) {
    const int tile_output_idx = tid + i * BLOCK_THREADS;
    if (FULL_TILE || tile_output_idx < num_left_outputs) {
      const int tile_output_seg_idx = tile_output_idx / EMBED_DIM;
      const int col_idx = tile_output_idx - tile_output_seg_idx * EMBED_DIM;
      const int param_seg_idx = s.indices[tile_output_seg_idx];
      const int param_idx = param_seg_idx * EMBED_DIM + col_idx;
      const int output_idx =
          tile_output_seg_idx * EMBED_DIM_SUM + embed_dim_offset + col_idx;
      d_tile_outputs[output_idx] = d_params[param_idx];
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT>
__device__ __forceinline__ void
GatherSegmentsCore(GatherSegmentsTempStorage<BLOCK_THREADS> &s,
                   const Tparam *__restrict__ d_params,
                   const AccessIndicesT &access_indices, const int num_inputs,
                   const int tile_idx, Tparam *__restrict__ d_outputs,
                   const int embed_dim_offset, const bool full_tile) {
  if (full_tile) {
    GatherSegmentsCore<true, EMBED_DIM_SUM, EMBED_DIM, UNROLL_FACTOR,
                       BLOCK_THREADS>(s, d_params, access_indices, num_inputs,
                                      tile_idx, d_outputs, embed_dim_offset);
  } else {
    GatherSegmentsCore<false, EMBED_DIM_SUM, EMBED_DIM, UNROLL_FACTOR,
                       BLOCK_THREADS>(s, d_params, access_indices, num_inputs,
                                      tile_idx, d_outputs, embed_dim_offset);
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT>
__device__ __forceinline__ void GatherSegments(
    GatherSegmentsTempStorage<BLOCK_THREADS> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    Tparam *__restrict__ d_outputs, const int num_inputs, const int task_bid,
    const int task_blocks, const int num_tiles, const int embed_dim_offset) {
  constexpr int TILE_SIZE = BLOCK_THREADS;

  for (int tile_idx = task_bid; tile_idx < num_tiles; tile_idx += task_blocks) {
    const int tile_offset = tile_idx * TILE_SIZE;
    const bool full_tile = tile_offset + TILE_SIZE <= num_inputs;

    GatherSegmentsCore<EMBED_DIM_SUM, EMBED_DIM, UNROLL_FACTOR, BLOCK_THREADS>(
        s, d_params, access_indices, num_inputs, tile_idx, d_outputs,
        embed_dim_offset, full_tile);

    __syncthreads();
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT>
__device__ __forceinline__ void
GatherSegmentsOneBlock(GatherSegmentsTempStorage<BLOCK_THREADS> &s,
                       const Tparam *__restrict__ d_params,
                       const AccessIndicesT &access_indices,
                       Tparam *__restrict__ d_outputs, const int num_inputs,
                       const int num_tiles, const int embed_dim_offset) {
  constexpr int TILE_SIZE = BLOCK_THREADS;

  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    const int tile_offset = tile_idx * TILE_SIZE;
    const bool full_tile = tile_offset + TILE_SIZE <= num_inputs;

    GatherSegmentsCore<EMBED_DIM_SUM, EMBED_DIM, UNROLL_FACTOR, BLOCK_THREADS>(
        s, d_params, access_indices, num_inputs, tile_idx, d_outputs,
        embed_dim_offset, full_tile);

    __syncthreads();
  }
}

template <int BLOCK_THREADS> struct GatherScatterSegmentsTempStorage {
  int indices[BLOCK_THREADS];
  int seg_ids[BLOCK_THREADS];
};

template <bool FULL_TILE, int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT,
          typename AccessSegmentIdT>
__device__ __forceinline__ void GatherScatterSegmentsCore(
    GatherScatterSegmentsTempStorage<BLOCK_THREADS> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegmentIdT &access_segment_ids, const int num_inputs,
    const int tile_idx, Tparam *__restrict__ d_outputs,
    const int embed_dim_offset) {
  constexpr int TILE_SIZE = BLOCK_THREADS;
  const int tile_offset = tile_idx * TILE_SIZE;
  const int tid = threadIdx.x;
  const int num_left_outputs = (num_inputs - tile_offset) * EMBED_DIM;

  const int input_idx = tile_offset + tid;
  if (FULL_TILE || input_idx < num_inputs) {
    s.indices[tid] = access_indices(input_idx);
    s.seg_ids[tid] = access_segment_ids(input_idx);
  }
  __syncthreads();

#pragma unroll UNROLL_FACTOR
  for (int i = 0; i < EMBED_DIM; ++i) {
    const int tile_output_idx = tid + i * BLOCK_THREADS;
    if (FULL_TILE || tile_output_idx < num_left_outputs) {
      const int tile_output_seg_idx = tile_output_idx / EMBED_DIM;
      const int col_idx = tile_output_idx - tile_output_seg_idx * EMBED_DIM;
      const int param_seg_idx = s.indices[tile_output_seg_idx];
      const int param_idx = param_seg_idx * EMBED_DIM + col_idx;
      const int output_seg_idx = s.seg_ids[tile_output_seg_idx];
      const int output_idx =
          output_seg_idx * EMBED_DIM_SUM + embed_dim_offset + col_idx;
      d_outputs[output_idx] = d_params[param_idx];
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT,
          typename AccessSegmentIdT>
__device__ __forceinline__ void GatherScatterSegmentsCore(
    GatherScatterSegmentsTempStorage<BLOCK_THREADS> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegmentIdT &access_segment_ids, const int num_inputs,
    const int tile_idx, Tparam *__restrict__ d_outputs,
    const int embed_dim_offset, const bool full_tile) {
  if (full_tile) {
    GatherScatterSegmentsCore<true, EMBED_DIM, UNROLL_FACTOR, BLOCK_THREADS>(
        s, d_params, access_indices, access_segment_ids, num_inputs, tile_idx,
        d_outputs, embed_dim_offset);
  } else {
    GatherScatterSegmentsCore<false, EMBED_DIM, UNROLL_FACTOR, BLOCK_THREADS>(
        s, d_params, access_indices, access_segment_ids, num_inputs, tile_idx,
        d_outputs, embed_dim_offset);
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int UNROLL_FACTOR,
          int BLOCK_THREADS, typename Tparam, typename AccessIndicesT,
          typename AccessSegmentIdT>
__device__ __forceinline__ void GatherScatterSegments(
    GatherScatterSegmentsTempStorage<BLOCK_THREADS> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegmentIdT &access_segment_ids,
    int *__restrict__ d_task_barriers, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int task_blocks, const int num_tiles, const int embed_dim_offset) {
  constexpr int TILE_SIZE = BLOCK_THREADS;
  for (int tile_idx = task_bid; tile_idx < num_tiles; tile_idx += task_blocks) {
    const int tile_offset = tile_idx * TILE_SIZE;
    const bool full_tile = tile_offset + TILE_SIZE <= num_inputs;

    GatherScatterSegmentsCore<EMBED_DIM, UNROLL_FACTOR, BLOCK_THREADS>(
        s, d_params, access_indices, access_segment_ids, num_inputs, tile_idx,
        d_outputs, embed_dim_offset, full_tile);

    __syncthreads();
  }
}

} // namespace onehot