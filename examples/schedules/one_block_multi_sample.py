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


class OneBlockMultiSampleParams(NamedTuple):
    vector_size: int
    items_per_thread: int
    vblock_dim_x: int
    vblock_dim_y: int


class OneBlockMultiSampleSchedule(ScheduleBase):
    def EmitScheduleBody(self, params: OneBlockMultiSampleParams) -> str:
        meta = self.meta
        vars_ = self.sched_vars
        vblock_dim_z = self.block_threads // (params.vblock_dim_x * params.vblock_dim_y)
        schedule_body = f'''
  const int num_inputs = {vars_.scala_args}[0];
  const int num_segments = {vars_.scala_args}[1];
  size_t offset = 0;
  auto d_indices = reinterpret_cast<const int *>({vars_.arg_buffer} + offset);
  offset += alignmem(num_inputs * sizeof(int));
  auto d_segment_offsets = reinterpret_cast<const int *>({vars_.arg_buffer} + offset);
  offset += alignmem((num_segments + 1) * sizeof(int));
  obms::SparseSegmentReduce<{vars_.EMBED_DIM_SUM}, {meta.embed_dim}, {params.vector_size}, {params.items_per_thread},
                            {params.vblock_dim_x}, {params.vblock_dim_y}, {vblock_dim_z}>(
      // s
      *reinterpret_cast<{self.EmitShmemType(params)} *>({vars_.shmem}),
      // d_params
      reinterpret_cast<const {meta.data_type} *>({vars_.embed_table}),
      // access_indices
      [&](int idx) {{ return d_indices[idx]; }},
      // access_segment_offsets
      [&](int idx) {{ return d_segment_offsets[idx]; }},
      // d_outputs
      reinterpret_cast<{meta.data_type} *>({vars_.output}),
      // num_inputs, num_segments
      num_inputs, num_segments,
      // task_bid
      {vars_.task_bid},
      // embed_dim_offset
      {vars_.embed_dim_offset});
'''
        return schedule_body

    def EmitShmemType(self, params: OneBlockMultiSampleParams) -> str:
        meta = self.meta
        vblock_dim_z = self.block_threads // (params.vblock_dim_x * params.vblock_dim_y)
        tparams = f"{params.vector_size}, {params.vblock_dim_x}, {params.vblock_dim_y}, {vblock_dim_z}, {meta.data_type}"
        return f"typename obms::SparseSegmentReduceTempStorage<{tparams}>"

    def EmitHostPreprocess(self, params: OneBlockMultiSampleParams) -> str:
        meta = self.meta
        vars_ = self.pre_vars
        vblock_dim_z = self.block_threads // (params.vblock_dim_x * params.vblock_dim_y)
        host = f'''
  obms::StringSplitToInt(
      // inputs
      {vars_.raw_data},
      // tile_size,
      {vblock_dim_z},
      // indices, segment_offsets,
      {vars_.data}[0], {vars_.data}[1],
      // num_tiles, num_blocks
      {vars_.num_tiles_ref}, {vars_.num_blocks_ref},
      // num_inputs, num_segments
      {vars_.scala_data}[0], {vars_.scala_data}[1]);
  {vars_.output_size_ref} = {vars_.scala_data}[1] * {meta.embed_dim} * sizeof({self.meta.data_type});
  {vars_.extra_buffer_size_ref} = 0;
'''
        return host

    def UsedArgCounts(self) -> Tuple[int, int]:
        # data[2] = indices, segment_offsets
        # scala_data[2] = num_inputs, num_segments
        return 2, 2

    def EmitHeaders(self) -> List[str]:
        return [
            "\"one_block_multi_sample_kernel.cuh\"",
        ]

    def EmitPrepropHeaders(self) -> List[str]:
        return [
            "<string>",
            "<vector>",
            "\"one_block_multi_sample_preprocess.cuh\"",
        ]

    def GenParamCandidates(self, stats: InputStats = None) -> List[OneBlockMultiSampleParams]:
        items_per_thread_candidates = [1, 2, 4, 8]
        vector_size_candidates = [1, 2, 4]
        vblock_dim_x_candidates = [1, 2, 4, 8, 16, 32, 64]
        vblock_dim_y_candidates = [4, 8, 16, 32, 64, 128]

        params = []
        for vblock_dim_x in vblock_dim_x_candidates:
            for vector_size in vector_size_candidates:
                if (vblock_dim_x * vector_size < 2 * self.meta.embed_dim
                    and vblock_dim_x * vector_size >= self.meta.embed_dim // 2):
                    for vblock_dim_y in vblock_dim_y_candidates:
                        vblock_dim_z = self.block_threads // (vblock_dim_x * vblock_dim_y)
                        if vblock_dim_z >= 1:
                            params.extend(
                                [
                                    OneBlockMultiSampleParams(vector_size=vector_size,
                                                            vblock_dim_x=vblock_dim_x,
                                                            vblock_dim_y=vblock_dim_y,
                                                            items_per_thread=items_per_thread)
                                    for items_per_thread in items_per_thread_candidates
                                ]
                            )
        return params