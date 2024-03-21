#pragma once
#include <cstdio>
#include <cub/cub.cuh>

#include "aligned_vector.cuh"

namespace obms {

template <int VECTOR_SIZE, int VBLOCK_DIM_X, int VBLOCK_DIM_Y, int VBLOCK_DIM_Z,
          typename Tparam>
struct SparseSegmentReduceTempStorage {
  AlignedVector<Tparam, VECTOR_SIZE>
      partial_reduction[VBLOCK_DIM_Z][VBLOCK_DIM_Y][VBLOCK_DIM_X];
};

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          int VBLOCK_DIM_Z, typename Tparam, typename AccessIndicesT,
          typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduceCore(
    SparseSegmentReduceTempStorage<VECTOR_SIZE, VBLOCK_DIM_X, VBLOCK_DIM_Y,
                                   VBLOCK_DIM_Z, Tparam> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int embed_dim_offset) {
  constexpr int VBLOCK_DIM_XY = VBLOCK_DIM_X * VBLOCK_DIM_Y;
  const int tid = threadIdx.x;
  const int tz = tid / VBLOCK_DIM_XY;
  const int txy = tid - tz * VBLOCK_DIM_XY;
  const int ty = txy / VBLOCK_DIM_X;
  const int tx = txy - ty * VBLOCK_DIM_X;
  const int segment_id = task_bid * VBLOCK_DIM_Z + tz;

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

  int begin, end;
  if (segment_id < num_segments) {
    begin = access_segoffsets(segment_id);
    end = access_segoffsets(segment_id + 1);
  }

#pragma unroll
  for (int x_offset = 0; x_offset < VECTORIZED_EMBED_DIM;
       x_offset += VBLOCK_DIM_X) {
    const int x = x_offset + tx;
    const bool x_ok = FULL_X || x < VECTORIZED_EMBED_DIM;

    Tvec thread_data(0);
    if (segment_id < num_segments) {
      for (int offset = begin; offset < end;
           offset += VBLOCK_DIM_Y * ITEMS_PER_THREAD) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
          const int y_idx = offset + ty * ITEMS_PER_THREAD + ITEM;
          const bool y_ok = y_idx < end;

          if (x_ok && y_ok) {
            const int lookup_idx = access_indices(y_idx);
            // separate load and add to enable memory coalescing
            auto params =
                d_vectorized_params[lookup_idx * VECTORIZED_EMBED_DIM + x];
            thread_data += params;
          }
        }
      }
    }

    constexpr bool WARP_XY = 32 % VBLOCK_DIM_XY == 0;
#pragma unroll
    for (int stride = (VBLOCK_DIM_Y + 1) / 2, last_stride = VBLOCK_DIM_Y;
         last_stride != stride; stride = ((last_stride = stride) + 1) / 2) {
      // TODO: use warp shuffle if 32 % VBLOCK_DIM_XY == 0
      if (x_ok && ty < last_stride) {
        s.partial_reduction[tz][ty][tx] = thread_data;
      }
      if (WARP_XY)
        __syncwarp();
      else
        __syncthreads();
      if (x_ok && ty + stride < last_stride) {
        thread_data += s.partial_reduction[tz][ty + stride][tx];
      }
      if (WARP_XY)
        __syncwarp();
      else
        __syncthreads();
    }

    if (segment_id < num_segments) {
      if (ty == 0 && x_ok) {
        auto d_vectorized_outputs = reinterpret_cast<Tvec *>(d_outputs);
        d_vectorized_outputs[segment_id * VECTORIZED_EMBED_DIM_SUM +
                             vectorized_embed_dim_offset + x] = thread_data;
      }
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int VECTOR_SIZE,
          int ITEMS_PER_THREAD, int VBLOCK_DIM_X, int VBLOCK_DIM_Y,
          int VBLOCK_DIM_Z, typename Tparam, typename AccessIndicesT,
          typename AccessSegoffsetsT>
__device__ __forceinline__ void SparseSegmentReduce(
    SparseSegmentReduceTempStorage<VECTOR_SIZE, VBLOCK_DIM_X, VBLOCK_DIM_Y,
                                   VBLOCK_DIM_Z, Tparam> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegoffsetsT &access_segoffsets, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments, const int task_bid,
    const int embed_dim_offset) {
  SparseSegmentReduceCore<EMBED_DIM_SUM, EMBED_DIM, VECTOR_SIZE,
                          ITEMS_PER_THREAD, VBLOCK_DIM_X, VBLOCK_DIM_Y,
                          VBLOCK_DIM_Z>(
      s, d_params, access_indices, access_segoffsets, d_outputs, num_inputs,
      num_segments, task_bid, embed_dim_offset);
}

}; // namespace obms