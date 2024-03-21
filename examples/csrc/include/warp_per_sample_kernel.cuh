#pragma once
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>

#include "aligned_vector.cuh"

namespace ws {

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename AccessIndicesT, typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduceCore(
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int embed_dim_offset) {
  const int tid = threadIdx.x;
  const int ty = tid / VBLOCK_DIM_X;
  const int tx = tid - ty * VBLOCK_DIM_X;
  const int segment_id = task_bid * VBLOCK_DIM_Y + ty;

  if (segment_id >= num_segments)
    return;

  static_assert(EMBED_DIM % VECTOR_SIZE == 0 &&
                    EMBED_DIM_SUM % VECTOR_SIZE == 0,
                "TODO: handle EMBED_DIM not divisible by VECTOR_SIZE");
  constexpr int VECTORIZED_EMBED_DIM = EMBED_DIM / VECTOR_SIZE;
  constexpr int VECTORIZED_EMBED_DIM_SUM = EMBED_DIM_SUM / VECTOR_SIZE;
  constexpr bool FULL_X = VECTORIZED_EMBED_DIM % VBLOCK_DIM_X == 0;
  using Tvec = AlignedVector<Tparam, VECTOR_SIZE>;
  auto *__restrict__ d_vectorized_params =
      reinterpret_cast<const Tvec *>(d_params);
  // TODO: if embed_dim_offset % VECTOR_SIZE != 0
  const int vectorized_embed_dim_offset = embed_dim_offset / VECTOR_SIZE;

  const int begin = access_segoffsets(segment_id);
  const int end = access_segoffsets(segment_id + 1);

#pragma unroll
  for (int x_offset = 0; x_offset < VECTORIZED_EMBED_DIM;
       x_offset += VBLOCK_DIM_X) {
    const int x = x_offset + tx;
    const bool x_ok = FULL_X || x < VECTORIZED_EMBED_DIM;

    Tvec thread_data(0);
    for (int offset = begin; offset < end; offset += ITEMS_PER_THREAD) {
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int lid = offset + ITEM;
        if (x_ok && lid < end) {
          const int lookup_idx = access_indices(lid);
          // separate load and add to enable memory coalescing
          auto params =
              d_vectorized_params[lookup_idx * VECTORIZED_EMBED_DIM + x];
          thread_data += params;
        }
      }
    }

    if (x_ok) {
      auto d_vectorized_outputs = reinterpret_cast<Tvec *>(d_outputs);
      d_vectorized_outputs[segment_id * VECTORIZED_EMBED_DIM_SUM +
                           vectorized_embed_dim_offset + x] = thread_data;
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          typename Tparam, typename AccessIndicesT, typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduce(
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int embed_dim_offset) {
  SparseSegmentReduceCore<EMBED_DIM_SUM, EMBED_DIM, VECTOR_SIZE,
                          ITEMS_PER_THREAD, VBLOCK_DIM_X, VBLOCK_DIM_Y>(
      d_params, access_indices, access_segoffsets, d_outputs, num_inputs,
      num_segments, task_bid, embed_dim_offset);
}

} // namespace ws