#pragma once
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include "task_sync.cuh"

namespace rbk {

#define RBKScanItemT ScanVecPair<SCAN_DIM, Tparam>
#define RBKScanOpT ScanVecScanOp<SCAN_DIM, Tparam>
#define RBKScanTileStateT cub::ScanTileState<RBKScanItemT>
#define RBKTilePrefixCallbackOpT                                               \
  cub::TilePrefixCallbackOp<RBKScanItemT, RBKScanOpT, RBKScanTileStateT>
#define RBKBlockScanT                                                          \
  cub::BlockScan<RBKScanItemT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>

template <typename T, bool SINGLE_WORD> struct GetAllocationSizeHelper;

template <typename T> struct GetAllocationSizeHelper<T, true> {
  __host__ __device__ size_t operator()(int num_tiles) {
    using StateT = cub::ScanTileState<T, true>;
    return (num_tiles + StateT::TILE_STATUS_PADDING) *
           sizeof(typename StateT::TileDescriptor);
  }
};

template <typename T> struct GetAllocationSizeHelper<T, false> {
  __host__ __device__ size_t operator()(int num_tiles) {
    using StateT = cub::ScanTileState<T, false>;

    size_t temp_storage_bytes;

    // Specify storage allocation requirements
    size_t allocation_sizes[3];
    // bytes needed for tile status descriptors
    allocation_sizes[0] = (num_tiles + StateT::TILE_STATUS_PADDING) *
                          sizeof(typename StateT::StatusWord);
    // bytes needed for partials
    allocation_sizes[1] = (num_tiles + StateT::TILE_STATUS_PADDING) *
                          sizeof(typename cub::Uninitialized<T>);
    // bytes needed for inclusives
    allocation_sizes[2] = (num_tiles + StateT::TILE_STATUS_PADDING) *
                          sizeof(typename cub::Uninitialized<T>);

    // Set the necessary size of the blob
    void *allocations[3] = {};
    cub::AliasTemporaries(NULL, temp_storage_bytes, allocations,
                          allocation_sizes);

    return temp_storage_bytes;
  }
};

template <typename T, bool SINGLE_WORD = cub::Traits<T>::PRIMITIVE>
__host__ __device__ __forceinline__ size_t GetAllocationSize(int num_tiles) {
  return GetAllocationSizeHelper<T, SINGLE_WORD>()(num_tiles);
}

template <typename T>
__device__ __forceinline__ void
InitializeStatus(cub::ScanTileState<T, false> &tile_status, int tile_idx,
                 int num_tiles) {
  using StateT = cub::ScanTileState<T, false>;
  using StatusEnum = cub::ScanTileStatus;

  const int tile_to_set = (tile_idx * blockDim.x) + threadIdx.x;
  if (tile_to_set < num_tiles) {
    // Not-yet-set
    tile_status.d_tile_status[StateT::TILE_STATUS_PADDING + tile_to_set] =
        StateT::StatusWord(StatusEnum::SCAN_TILE_INVALID);
  }

  if ((tile_idx == 0) && (threadIdx.x < StateT::TILE_STATUS_PADDING)) {
    // Padding
    tile_status.d_tile_status[threadIdx.x] =
        StateT::StatusWord(StatusEnum::SCAN_TILE_OOB);
  }
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ void
InitializeStatus(cub::ScanTileState<T, false> &tile_status, int task_bid,
                 int task_blocks, int num_tiles) {
  using StateT = cub::ScanTileState<T, false>;
  using StatusEnum = cub::ScanTileStatus;

  for (int tile_to_set = (task_bid * BLOCK_THREADS) + threadIdx.x;
       tile_to_set < num_tiles; tile_to_set += task_blocks * BLOCK_THREADS) {
    // Not-yet-set
    tile_status.d_tile_status[StateT::TILE_STATUS_PADDING + tile_to_set] =
        StateT::StatusWord(StatusEnum::SCAN_TILE_INVALID);
  }

  if ((task_bid == 0) && (threadIdx.x < StateT::TILE_STATUS_PADDING)) {
    // Padding
    tile_status.d_tile_status[threadIdx.x] =
        StateT::StatusWord(StatusEnum::SCAN_TILE_OOB);
  }
}

template <typename T>
__device__ __forceinline__ void
InitializeStatus(cub::ScanTileState<T, true> &tile_status, int tile_idx,
                 int num_tiles) {
  using StateT = cub::ScanTileState<T, true>;
  using StatusEnum = cub::ScanTileStatus;

  const int tile_to_set = (tile_idx * blockDim.x) + threadIdx.x;
  typename StateT::TxnWord val;
  auto descriptor = reinterpret_cast<typename StateT::TileDescriptor *>(&val);

  if (tile_to_set < num_tiles) {
    // Not-yet-set
    descriptor->status = StateT::StatusWord(StatusEnum::SCAN_TILE_INVALID);
    tile_status.d_tile_descriptors[StateT::TILE_STATUS_PADDING + tile_to_set] =
        val;
  }

  if ((tile_idx == 0) && (threadIdx.x < StateT::TILE_STATUS_PADDING)) {
    // Padding
    descriptor->status = StateT::StatusWord(StatusEnum::SCAN_TILE_OOB);
    tile_status.d_tile_descriptors[threadIdx.x] = val;
  }
}

template <int SCAN_DIM, typename Tparam> struct ScanVecPair {
  Tparam scan_vec[SCAN_DIM];
  int scan_key;

  __host__ __device__ __forceinline__ ScanVecPair<SCAN_DIM, Tparam>() = default;

  __host__ __device__ __forceinline__
  ScanVecPair<SCAN_DIM, Tparam>(const ScanVecPair<SCAN_DIM, Tparam> &rhv)
      : scan_key(rhv.scan_key) {
#pragma unroll
    for (int i = 0; i < SCAN_DIM; ++i) {
      scan_vec[i] = rhv.scan_vec[i];
    }
  }

  __host__ __device__ __forceinline__ ScanVecPair<SCAN_DIM, Tparam> &
  operator=(const ScanVecPair<SCAN_DIM, Tparam> &rhv) {
    if (&rhv == this) {
      return *this;
    }
    scan_key = rhv.scan_key;
#pragma unroll
    for (int i = 0; i < SCAN_DIM; ++i) {
      scan_vec[i] = rhv.scan_vec[i];
    }
    return *this;
  }
};

template <int SCAN_DIM, typename Tparam> struct ScanVecScanOp {
  __device__ __forceinline__ ScanVecPair<SCAN_DIM, Tparam>
  operator()(const ScanVecPair<SCAN_DIM, Tparam> &first,
             const ScanVecPair<SCAN_DIM, Tparam> &second) {
    ScanVecPair<SCAN_DIM, Tparam> ret;
    ret.scan_key = first.scan_key + second.scan_key;
    const bool flag = second.scan_key;
#pragma unroll
    for (int i = 0; i < SCAN_DIM; ++i) {
      float adder = flag ? 0 : first.scan_vec[i];
      ret.scan_vec[i] = adder + second.scan_vec[i];
    }

    return ret;
  }
};

template <int SCAN_DIM, int BLOCK_THREADS, typename Tparam>
struct SparseSegmentSumTempStorage {
  union {
    struct {
      typename RBKBlockScanT::TempStorage scan;
      typename RBKTilePrefixCallbackOpT::TempStorage prefix;
    } scan_storage;
    int seg_ids[BLOCK_THREADS];
  };
};

template <int EMBED_DIM, int SCAN_DIM, int BLOCK_THREADS, typename Tparam>
union SparseSegmentSumTempStorageWrapper {
  SparseSegmentSumTempStorage<SCAN_DIM, BLOCK_THREADS, Tparam> normal;
  SparseSegmentSumTempStorage<EMBED_DIM % SCAN_DIM, BLOCK_THREADS, Tparam> left;
};

template <bool IS_LAST_TILE, int EMBED_DIM_SUM, int EMBED_DIM, int SCAN_DIM,
          int ITEMS_PER_THREAD, int BLOCK_THREADS, typename Tparam,
          typename AccessSegmentIdT, typename AccessIndicesT>
__device__ __forceinline__ void SparseSegmentSumCore(
    SparseSegmentSumTempStorage<SCAN_DIM, BLOCK_THREADS, Tparam> &s,
    const Tparam *__restrict__ d_params,
    const AccessSegmentIdT &access_segment_ids,
    const AccessIndicesT &access_indices, const int embed_offset,
    cub::ScanTileState<ScanVecPair<SCAN_DIM, Tparam>> &tile_state,
    const int tile_idx, const int num_inputs, Tparam *__restrict__ d_outputs,
    const int embed_dim_offset) {
  constexpr int TILE_SIZE = ITEMS_PER_THREAD * BLOCK_THREADS;
  const int tile_offset = tile_idx * TILE_SIZE;
  const int tid = threadIdx.x;
  int indices[ITEMS_PER_THREAD];
  int seg_ids[ITEMS_PER_THREAD + 1];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int idx = tile_offset + tid * ITEMS_PER_THREAD + i;
    if (!IS_LAST_TILE || idx < num_inputs) {
      indices[i] = access_indices(idx);
      seg_ids[i + 1] = access_segment_ids(idx);
    } else {
      seg_ids[i + 1] = -1;
    }
  }

  // store segment id
  s.seg_ids[tid] = seg_ids[ITEMS_PER_THREAD];
  __syncthreads();

  seg_ids[0] = (tid > 0)         ? s.seg_ids[tid - 1]
               : (tile_idx == 0) ? -1
                                 : access_segment_ids(tile_offset - 1);
  __syncthreads();

  RBKScanItemT scan_items[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    scan_items[i].scan_key = seg_ids[i + 1] - seg_ids[i];
    const int idx = tile_offset + tid * ITEMS_PER_THREAD + i;
    if (!IS_LAST_TILE || idx < num_inputs) {
#pragma unroll
      for (int j = 0; j < SCAN_DIM; ++j) {
        // lookup and store to registers
        scan_items[i].scan_vec[j] =
            d_params[indices[i] * EMBED_DIM + embed_offset + j];
      }
    }
  }

  auto store_item = [&](const RBKScanItemT &item, const int &out_seg) {
#pragma unroll
    for (int i = 0; i < SCAN_DIM; ++i) {
      d_outputs[out_seg * EMBED_DIM_SUM + embed_dim_offset + embed_offset + i] =
          item.scan_vec[i];
    }
  };

  // scan
  RBKScanOpT scan_op;
  if (tile_idx == 0) {
    RBKScanItemT block_aggregate;
    RBKBlockScanT(s.scan_storage.scan)
        .ExclusiveScan(scan_items, scan_items, scan_op, block_aggregate);

    if ((!IS_LAST_TILE) && (tid == 0)) {
      tile_state.SetInclusive(0, block_aggregate);
    }

    // if last tile is full (seg_id >= 0)
    if (IS_LAST_TILE && tid + 1 == BLOCK_THREADS &&
        seg_ids[ITEMS_PER_THREAD] >= 0) {
      store_item(block_aggregate, seg_ids[ITEMS_PER_THREAD]);
    }
  } else {
    // scan non-first tile
    RBKTilePrefixCallbackOpT prefix_op(tile_state, s.scan_storage.prefix,
                                       scan_op, tile_idx);
    RBKBlockScanT(s.scan_storage.scan)
        .ExclusiveScan(scan_items, scan_items, scan_op, prefix_op);

    // if last tile is full (seg_id >= 0)
    if (IS_LAST_TILE && tid + 1 == BLOCK_THREADS &&
        seg_ids[ITEMS_PER_THREAD] >= 0) {
      RBKScanItemT total_aggregate = prefix_op.GetInclusivePrefix();
      store_item(total_aggregate, seg_ids[ITEMS_PER_THREAD]);
    }
  }

  // store to global memory
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (seg_ids[i + 1] - seg_ids[i] && seg_ids[i] >= 0) {
      store_item(scan_items[i], seg_ids[i]);
    }
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int SCAN_DIM, int ITEMS_PER_THREAD,
          int BLOCK_THREADS, typename Tparam, typename AccessSegmentIdT,
          typename AccessIndicesT>
__device__ __forceinline__ void SparseSegmentSumCore(
    SparseSegmentSumTempStorage<SCAN_DIM, BLOCK_THREADS, Tparam> &s,
    const Tparam *__restrict__ d_params,
    const AccessSegmentIdT &access_segment_ids,
    const AccessIndicesT &access_indices, const int embed_offset,
    cub::ScanTileState<ScanVecPair<SCAN_DIM, Tparam>> &tile_state,
    const int tile_idx, const int num_inputs, Tparam *__restrict__ d_outputs,
    const int embed_dim_offset, const bool is_last_tile) {
  if (is_last_tile) {
    SparseSegmentSumCore<true, EMBED_DIM_SUM, EMBED_DIM, SCAN_DIM,
                         ITEMS_PER_THREAD, BLOCK_THREADS>(
        s, d_params, access_segment_ids, access_indices, embed_offset,
        tile_state, tile_idx, num_inputs, d_outputs, embed_dim_offset);
  } else {
    SparseSegmentSumCore<false, EMBED_DIM_SUM, EMBED_DIM, SCAN_DIM,
                         ITEMS_PER_THREAD, BLOCK_THREADS>(
        s, d_params, access_segment_ids, access_indices, embed_offset,
        tile_state, tile_idx, num_inputs, d_outputs, embed_dim_offset);
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int SCAN_DIM, int ITEMS_PER_THREAD,
          int BLOCK_THREADS, typename Tparam, typename AccessSegmentIdT,
          typename AccessIndicesT>
__device__ __forceinline__ void SparseSegmentSum(
    SparseSegmentSumTempStorageWrapper<EMBED_DIM, SCAN_DIM, BLOCK_THREADS,
                                       Tparam> &s,
    const Tparam *__restrict__ d_params, const AccessIndicesT &access_indices,
    const AccessSegmentIdT &access_segment_ids,
    int *__restrict__ d_task_barriers, char *__restrict__ d_tile_status_storage,
    Tparam *__restrict__ d_outputs, const int num_inputs,
    const int num_segments, const int tile_status_storage_bytes,
    const int task_bid, const int task_blocks, const int num_tiles,
    const int embed_dim_offset) {
  constexpr int TILE_SIZE = ITEMS_PER_THREAD * BLOCK_THREADS;

  RBKScanTileStateT tile_state;
  tile_state.Init(num_tiles, d_tile_status_storage, tile_status_storage_bytes);

  int embed_offset;
#pragma unroll 1
  for (embed_offset = 0; embed_offset < EMBED_DIM; embed_offset += SCAN_DIM) {
    InitializeStatus<BLOCK_THREADS>(tile_state, task_bid, task_blocks,
                                    num_tiles);

    task_sync<BLOCK_THREADS>(d_task_barriers + blockIdx.x - task_bid, task_bid,
                             task_blocks);

    // TODO: try using one block to process sequent tiles to reduce the times of
    // lookback
    for (int tile_idx = task_bid; tile_idx < num_tiles;
         tile_idx += task_blocks) {
      const int tile_offset = tile_idx * TILE_SIZE;
      const bool is_last_tile = tile_offset + TILE_SIZE >= num_inputs;

      SparseSegmentSumCore<EMBED_DIM_SUM, EMBED_DIM, SCAN_DIM, ITEMS_PER_THREAD,
                           BLOCK_THREADS>(
          s.normal, d_params, access_segment_ids, access_indices, embed_offset,
          tile_state, tile_idx, num_inputs, d_outputs, embed_dim_offset,
          is_last_tile);

      __syncthreads();
    }

    task_sync<BLOCK_THREADS>(d_task_barriers + blockIdx.x - task_bid, task_bid,
                             task_blocks);
  }

  if (EMBED_DIM % SCAN_DIM != 0) {
    static_assert(
        EMBED_DIM % SCAN_DIM == 0,
        "TODO: support conditions that EMBED_DIM not divisible by SCAN_DIM");
  }
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int SCAN_DIM, int ITEMS_PER_THREAD,
          int BLOCK_THREADS, typename Tparam, typename Tsegment,
          typename Tindice>
__global__ void SparseSegmentSumKernel(
    const Tparam *__restrict__ d_params,
    const Tsegment *__restrict__ d_segment_ids,
    const Tindice *__restrict__ d_indices,
    const int2 *__restrict__ d_task_mapping,
    const int *__restrict__ d_task_metas, int *__restrict__ d_task_barriers,
    char *__restrict__ d_tile_status_storage, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments,
    const int tile_status_storage_bytes, const int max_concurrent_blocks,
    const int embed_dim_offset) {
  const int2 task = d_task_mapping[blockIdx.x];
  const int feat_idx = task.x;
  const int task_bid = task.y;
  const int num_tiles = d_task_metas[feat_idx];
  const int task_blocks = min(num_tiles, max_concurrent_blocks);

  auto access_segment_ids = [&](int idx) { return d_segment_ids[idx]; };
  auto access_indices = [&](int idx) { return d_indices[idx]; };

  __shared__ SparseSegmentSumTempStorageWrapper<EMBED_DIM, SCAN_DIM,
                                                BLOCK_THREADS, Tparam>
      s;

  SparseSegmentSum<EMBED_DIM_SUM, EMBED_DIM, SCAN_DIM, ITEMS_PER_THREAD,
                   BLOCK_THREADS>(
      s, d_params, access_indices, access_segment_ids, d_task_barriers,
      d_tile_status_storage, d_outputs, num_inputs, num_segments,
      tile_status_storage_bytes, task_bid, task_blocks, num_tiles,
      embed_dim_offset);
}

template <int EMBED_DIM_SUM, int EMBED_DIM, int SCAN_DIM,
          int ITEMS_PER_THREAD, int BLOCK_THREADS, typename Tparam,
          typename Tsegment, typename Tindice>
void SparseSegmentSumLauncher(
    const Tparam *__restrict__ d_params,
    const Tsegment *__restrict__ d_segment_ids,
    const Tindice *__restrict__ d_indices,
    const int2 *__restrict__ d_task_mapping,
    const int *__restrict__ d_task_metas, int *__restrict__ d_task_barriers,
    char *__restrict__ d_tile_status_storage, Tparam *__restrict__ d_outputs,
    const int num_inputs, const int num_segments,
    const int tile_status_storage_bytes, const int max_concurrent_blocks,
    const int task_block_sum, const int embed_dim_offset) {
  SparseSegmentSumKernel<EMBED_DIM_SUM, EMBED_DIM, SCAN_DIM, ITEMS_PER_THREAD,
                         BLOCK_THREADS, Tparam, Tsegment, Tindice>
      <<<task_block_sum, BLOCK_THREADS>>>(
          d_params, d_segment_ids, d_indices, d_task_mapping, d_task_metas,
          d_task_barriers, d_tile_status_storage, d_outputs, num_inputs,
          num_segments, tile_status_storage_bytes, max_concurrent_blocks,
          embed_dim_offset);
}

#undef RBKScanItemT
#undef RBKScanOpT
#undef RBKScanTileStateT
#undef RBKTilePrefixCallbackOpT
#undef RBKBlockScanT

} // namespace rbk