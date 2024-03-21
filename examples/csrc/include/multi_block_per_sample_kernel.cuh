#pragma once
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>

#include "aligned_vector.cuh"
#include "task_sync.cuh"

namespace mbs {

template <int VECTOR_SIZE, int VBLOCK_DIM_X, int VBLOCK_DIM_Y, typename Tparam>
struct SparseSegmentReduceTempStorage {
  AlignedVector<Tparam, VECTOR_SIZE> partial_reduction[VBLOCK_DIM_Y]
                                                      [VBLOCK_DIM_X];
};

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename AccessIndicesT, typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduceCore(
    SparseSegmentReduceTempStorage<VECTOR_SIZE, VBLOCK_DIM_X, VBLOCK_DIM_Y,
                                   Tparam> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int segment_id, const int work_offset,
    const int embed_dim_offset) {
  const int tid = threadIdx.x;
  const int ty = tid / VBLOCK_DIM_X;
  const int tx = tid - ty * VBLOCK_DIM_X;

  static_assert(EMBED_DIM % VECTOR_SIZE == 0,
                "TODO: handle EMBED_DIM not divisible by VECTOR_SIZE");
  constexpr int VECTORIZED_EMBED_DIM = EMBED_DIM / VECTOR_SIZE;
  constexpr bool FULL_X = VECTORIZED_EMBED_DIM % VBLOCK_DIM_X == 0;
  using Tvec = AlignedVector<Tparam, VECTOR_SIZE>;
  auto *__restrict__ d_vectorized_params =
      reinterpret_cast<const Tvec *>(d_params);
  const int begin = access_segoffsets(segment_id);
  const int end = access_segoffsets(segment_id + 1);

#pragma unroll
  for (int x_offset = 0; x_offset < VECTORIZED_EMBED_DIM;
       x_offset += VBLOCK_DIM_X) {
    const int x = x_offset + tx;
    const bool x_ok = FULL_X || x < VECTORIZED_EMBED_DIM;

    Tvec thread_data(0);
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      const int y_idx = begin + work_offset + ty * ITEMS_PER_THREAD + ITEM;
      const bool y_ok = y_idx < end;

      if (x_ok && y_ok) {
        const int lookup_idx = access_indices(y_idx);
        // separate load and add to enable memory coalescing
        auto params =
            d_vectorized_params[lookup_idx * VECTORIZED_EMBED_DIM + x];
        thread_data += params;
      }
    }

#pragma unroll
    for (int stride = (VBLOCK_DIM_Y + 1) / 2, last_stride = VBLOCK_DIM_Y;
         last_stride != stride; stride = ((last_stride = stride) + 1) / 2) {
      if (x_ok && ty < last_stride) {
        s.partial_reduction[ty][tx] = thread_data;
      }
      __syncthreads();
      if (x_ok && ty + stride < last_stride) {
        thread_data += s.partial_reduction[ty + stride][tx];
      }
      __syncthreads();
    }

    if (ty == 0 && x_ok) {
      const int offset =
          segment_id * EMBED_DIM_SUM + embed_dim_offset + x * VECTOR_SIZE;
#pragma unroll
      for (int VECTOR_ITEM = 0; VECTOR_ITEM < VECTOR_SIZE; ++VECTOR_ITEM) {
        atomicAdd(&d_outputs[offset + VECTOR_ITEM], thread_data[VECTOR_ITEM]);
      }
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename AccessIndicesT, typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduce(
    SparseSegmentReduceTempStorage<VECTOR_SIZE, VBLOCK_DIM_X, VBLOCK_DIM_Y,
                                   Tparam> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets,
    int *__restrict__ d_task_barriers,
    const int2 *__restrict__ d_workload_mapping, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int task_blocks, const int num_tiles, const int embed_dim_offset) {
  constexpr int BLOCK_THREADS = VBLOCK_DIM_X * VBLOCK_DIM_Y;

  for (int tile_idx = task_bid; tile_idx < num_tiles; tile_idx += task_blocks) {
    int2 work_info = d_workload_mapping[tile_idx];
    int segment_id = work_info.x;
    int work_offset = work_info.y * ITEMS_PER_THREAD * VBLOCK_DIM_Y;

    SparseSegmentReduceCore<EMBED_DIM_SUM, EMBED_DIM, VECTOR_SIZE,
                            ITEMS_PER_THREAD, VBLOCK_DIM_X, VBLOCK_DIM_Y>(
        s, d_params, access_indices, access_segoffsets, d_outputs, num_inputs,
        segment_id, work_offset, embed_dim_offset);

    __syncthreads();
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename Tindices, typename Tsegoffsets>
__global__ void SparseSegmentSumKernel(
    const Tparam *__restrict__ d_params, const Tindices *__restrict__ d_indices,
    const Tsegoffsets *__restrict__ d_segment_offsets,
    const int2 *__restrict__ d_task_mapping,
    const int *__restrict__ d_task_metas, int *__restrict__ d_task_barriers,
    const int2 *__restrict__ d_workload_mapping, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int embed_dim_offset,
    const int max_concurrent_blocks) {
  const int2 task = d_task_mapping[blockIdx.x];
  const int feat_idx = task.x;
  const int task_bid = task.y;
  const int num_tiles = d_task_metas[feat_idx];
  const int task_blocks = min(num_tiles, max_concurrent_blocks);

  auto access_segoffsets = [&](int idx) { return d_segment_offsets[idx]; };
  auto access_indices = [&](int idx) { return d_indices[idx]; };

  __shared__ SparseSegmentReduceTempStorage<VECTOR_SIZE, VBLOCK_DIM_X,
                                            VBLOCK_DIM_Y, Tparam>
      s;

  SparseSegmentReduce<EMBED_DIM_SUM, EMBED_DIM, VECTOR_SIZE, ITEMS_PER_THREAD,
                      VBLOCK_DIM_X, VBLOCK_DIM_Y>(
      s, d_params, access_indices, access_segoffsets, d_task_barriers,
      d_workload_mapping, d_outputs, num_inputs, num_segments, task_bid,
      task_blocks, num_tiles, embed_dim_offset);
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename Tindices, typename Tsegoffsets>
void SparseSegmentSumLauncher(
    const Tparam *__restrict__ d_params, const Tindices *__restrict__ d_indices,
    const Tsegoffsets *__restrict__ d_segment_offsets,
    const int2 *__restrict__ d_task_mapping,
    const int *__restrict__ d_task_metas, int *__restrict__ d_task_barriers,
    const int2 *__restrict__ d_workload_mapping, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int embed_dim_offset,
    const int max_concurrent_blocks, const int task_block_sum) {
  SparseSegmentSumKernel<EMBED_DIM_SUM, EMBED_DIM, VECTOR_SIZE,
                         ITEMS_PER_THREAD, VBLOCK_DIM_X, VBLOCK_DIM_Y>
      <<<task_block_sum, VBLOCK_DIM_X * VBLOCK_DIM_Y>>>(
          d_params, d_indices, d_segment_offsets, d_task_mapping, d_task_metas,
          d_task_barriers, d_workload_mapping, d_outputs, num_inputs,
          num_segments, embed_dim_offset, max_concurrent_blocks);
}

} // namespace mbs