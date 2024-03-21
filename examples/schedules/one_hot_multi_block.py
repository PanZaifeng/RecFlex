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


class OneHotMultiBlockParams(NamedTuple):
    unroll_factor: int


class OneHotMultiBlockSchedule(ScheduleBase):
    def EmitScheduleBody(self, params: OneHotMultiBlockParams) -> str:
        meta = self.meta
        vars_ = self.sched_vars
        schedule_body = f'''
  const int num_inputs = {vars_.scala_args}[0];
  auto d_indices = reinterpret_cast<const int *>({vars_.arg_buffer});
  onehot::GatherSegments<{vars_.EMBED_DIM_SUM}, {meta.embed_dim}, {params.unroll_factor}, {self.block_threads}>(
      // s
      *reinterpret_cast<{self.EmitShmemType(params)} *>({vars_.shmem}),
      // d_params
      reinterpret_cast<const {meta.data_type} *>({vars_.embed_table}),
      // access_indices
      [&](int idx) {{ return d_indices[idx]; }},
      // d_outputs
      reinterpret_cast<{meta.data_type} *>({vars_.output}),
      // num_inputs
      num_inputs,
      // task_bid, task_blocks, num_tiles
      {vars_.task_bid}, {vars_.task_blocks}, {vars_.num_tiles},
      // embed_dim_offset
      {vars_.embed_dim_offset});
'''
        return schedule_body

    def EmitShmemType(self, params: OneHotMultiBlockParams) -> str:
        return f"typename onehot::GatherSegmentsTempStorage<{self.block_threads}>"

    def EmitHostPreprocess(self, params: OneHotMultiBlockParams) -> str:
        meta = self.meta
        vars_ = self.pre_vars
        host = f'''
  onehot::OneHotMultiBlock(
      // inputs
      {vars_.raw_data},
      // max_concurrent_blocks
      {vars_.max_concurrent_blocks},
      // tile_size
      {self.block_threads},
      // indices
      {vars_.data}[0],
      // num_tiles, num_blocks
      {vars_.num_tiles_ref}, {vars_.num_blocks_ref},
      // num_inputs
      {vars_.scala_data}[0]);
  {vars_.output_size_ref} = {vars_.scala_data}[0] * {meta.embed_dim} * sizeof({self.meta.data_type});
  {vars_.extra_buffer_size_ref} = 0;
'''
        return host

    def UsedArgCounts(self) -> Tuple[int, int]:
        # data[1] = indices
        # scala_data[1] = num_inputs
        return 1, 1

    def EmitHeaders(self) -> List[str]:
        return [
            "\"onehot_kernel.cuh\"",
        ]

    def EmitPrepropHeaders(self) -> List[str]:
        return [
            "<string>",
            "<vector>",
            "\"onehot_preprocess.cuh\"",
        ]

    def GenParamCandidates(self, stats: InputStats = None) -> List[OneHotMultiBlockParams]:
        unroll_factor_candidates = divisors(self.meta.embed_dim)
        return [OneHotMultiBlockParams(unroll_factor=unroll_factor)
                for unroll_factor in unroll_factor_candidates]
