# Copyright 2024 The RecFlex Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from RecFlex.code_emitter import ScheduleBase, InputStats

from typing import List, NamedTuple, Tuple
from sympy import divisors


class ReduceByKeyParams(NamedTuple):
    scan_dim: int
    items_per_thread: int


class ReduceByKeySchedule(ScheduleBase):
    def EmitScheduleBody(self, params: ReduceByKeyParams) -> str:
        meta = self.meta
        vars_ = self.sched_vars
        schedule_body = f'''
  const int num_inputs = {vars_.scala_args}[0];
  const int num_segments = {vars_.scala_args}[1];
  const int tile_status_storage_bytes = {vars_.scala_args}[2];
  auto d_indices = reinterpret_cast<const int *>({vars_.arg_buffer});
  auto d_segment_ids = reinterpret_cast<const int *>({vars_.arg_buffer} + alignmem(num_inputs * sizeof(int)));
  rbk::SparseSegmentSum<{vars_.EMBED_DIM_SUM}, {meta.embed_dim}, {params.scan_dim}, {params.items_per_thread}, {self.block_threads}>(
      // s
      *reinterpret_cast<{self.EmitShmemType(params)} *>({vars_.shmem}),
      // d_params
      reinterpret_cast<const {meta.data_type} *>({vars_.embed_table}),
      // access_indices
      [&](int idx) {{ return d_indices[idx]; }},
      // access_segment_ids
      [&](int idx) {{ return d_segment_ids[idx]; }},
      // d_task_barriers
      {vars_.task_barriers},
      // d_tile_status_storage
      reinterpret_cast<char *>({vars_.extra_buffer}),
      // d_outputs
      reinterpret_cast<{meta.data_type} *>({vars_.output}),
      // num_inputs, num_segments, tile_status_storage_bytes
      num_inputs, num_segments, tile_status_storage_bytes,
      // task_bid, task_blocks, num_tiles
      {vars_.task_bid}, {vars_.task_blocks}, {vars_.num_tiles},
      // embed_dim_offset
      {vars_.embed_dim_offset});
'''
        return schedule_body

    def EmitShmemType(self, params: ReduceByKeyParams) -> str:
        meta = self.meta
        tparams = f"{meta.embed_dim}, {params.scan_dim}, {self.block_threads}, {meta.data_type}"
        return f"typename rbk::SparseSegmentSumTempStorageWrapper<{tparams}>"

    def EmitHostPreprocess(self, params: ReduceByKeyParams) -> str:
        meta = self.meta
        vars_ = self.pre_vars
        host = f'''
  rbk::StringSplitToInt<{params.scan_dim}, {meta.data_type}>(
      // inputs
      {vars_.raw_data},
      // max_concurrent_blocks
      {vars_.max_concurrent_blocks},
      // tile_size
      {self.block_threads * params.items_per_thread},
      // indices, segment_ids
      {vars_.data}[0], {vars_.data}[1],
      // num_tiles, num_blocks
      {vars_.num_tiles_ref}, {vars_.num_blocks_ref},
      // num_inputs, num_segments, tile_status_storage_bytes
      {vars_.scala_data}[0], {vars_.scala_data}[1], {vars_.scala_data}[2]);
  {vars_.output_size_ref} = {vars_.scala_data}[1] * {meta.embed_dim} * sizeof({self.meta.data_type});
  {vars_.extra_buffer_size_ref} = {vars_.scala_data}[2];
'''
        return host

    def UsedArgCounts(self) -> Tuple[int, int]:
        # data[2] = indices, segment_ids
        # scala_data[3] = num_inputs, num_segments, tile_status_storage_bytes
        return 2, 3

    def EmitHeaders(self) -> List[str]:
        return [
            "\"reduce_by_key_kernel.cuh\"",
        ]

    def EmitPrepropHeaders(self) -> List[str]:
        return [
            "<string>",
            "<vector>",
            "\"reduce_by_key_preprocess.cuh\"",
        ]

    def GenParamCandidates(self, stats: InputStats = None) -> List[ReduceByKeyParams]:
        if self.meta.embed_dim <= 32:
            items_per_thread_candidates = [1, 2, 4, 8]
            scan_dim_candidates = divisors(self.meta.embed_dim)
            return [
                ReduceByKeyParams(scan_dim=scan_dim, items_per_thread=items_per_thread)
                for items_per_thread in items_per_thread_candidates
                for scan_dim in scan_dim_candidates
            ]
        else:
            return []