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

import torch
import torch.nn as nn

from typing import List, Tuple
from types import ModuleType


class FeatPreprocess(nn.Module):
    def __init__(self, recom_host: ModuleType, device: torch.device):
        super(FeatPreprocess, self).__init__()
        self.recom_host = recom_host
        self.device = device

    def forward(self, raw_data: List[List[str]], max_concurrent_blocks: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                      torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        data_buffer_offsets, output_offsets, extra_buffer_offsets, task_mapping, task_tiles, data, scala_data = self.recom_host.preprocess(raw_data, max_concurrent_blocks)

        output_size: int = output_offsets[-1]
        total_blocks = len(task_mapping)

        extra_buffers = torch.empty([extra_buffer_offsets[-1]], dtype=torch.int8, device=self.device)
        task_barriers = torch.zeros([total_blocks], dtype=torch.int32, device=self.device)

        data_buffer_offsets = torch.tensor(data_buffer_offsets, dtype=torch.int32, device=self.device)
        extra_buffer_offsets = torch.tensor(extra_buffer_offsets, dtype=torch.int32, device=self.device)
        task_mapping = torch.tensor(task_mapping, dtype=torch.int32, device=self.device)
        task_tiles = torch.tensor(task_tiles, dtype=torch.int32, device=self.device)

        flat_data = [item for sublist in data for item in sublist]
        arg_buffers = torch.tensor(flat_data, dtype=torch.int32, device=self.device)
        scala_args = torch.tensor(scala_data, dtype=torch.int32, device=self.device)

        return (
            arg_buffers,
            data_buffer_offsets,
            scala_args,
            extra_buffers,
            extra_buffer_offsets,
            task_mapping,
            task_tiles,
            task_barriers,
            output_size,
        )


class FeatEmbed(nn.Module):
    def __init__(self, device: torch.device, embed_tables: List[torch.Tensor] = None, table_shapes: List[Tuple[int, int]] = None):
        super(FeatEmbed, self).__init__()
        self.device = device

        if embed_tables:
            self.embed_tables = embed_tables
        elif table_shapes:
            self.embed_tables = []
            for table_shape in table_shapes:
                embed_table = nn.Parameter(torch.randn(table_shape, dtype=torch.float32, device=self.device))
                self.embed_tables.append(embed_table.data)
        else:
            raise Exception("At least one of embed_tables and table_shapes should be given!")
        self.embed_table_ptrs = torch.ops.recom.get_gpu_pointers_array(self.embed_tables)
        self.max_concurrent_blocks = torch.ops.recom.get_max_concurrent_blocks()

        self.output_units = 0
        for embed_table in self.embed_tables:
            self.output_units += embed_table.shape[-1]

    def forward(self, arg_buffers: torch.Tensor, data_buffer_offsets: torch.Tensor,
                scala_args: torch.Tensor, extra_buffers: torch.Tensor, extra_buffer_offsets: torch.Tensor,
                task_mapping: torch.Tensor, task_tiles: torch.Tensor, task_barriers: torch.Tensor,
                output_size: int, timing: bool = False, timing_schedules: int = None):
        result = torch.ops.recom.process(
            self.embed_table_ptrs,
            arg_buffers,
            data_buffer_offsets,
            scala_args,
            extra_buffers,
            extra_buffer_offsets,
            task_mapping,
            task_tiles,
            task_barriers,
            output_size,
            self.max_concurrent_blocks
        )

        if timing:
            if not timing_schedules or timing_schedules > len(self.embed_tables):
                timing_schedules = len(self.embed_tables)
            output, times = result
            task_times = torch.zeros([timing_schedules], dtype=torch.float64)
            task_mapping = task_mapping.cpu()
            times = times.cpu()
            block_idx = 0
            total_blocks = task_mapping.shape[0]
            for feat_id in range(timing_schedules):
                while block_idx < total_blocks and task_mapping[block_idx, 0] == feat_id:
                    task_times[feat_id] += times[block_idx]
                    block_idx += 1
            return output, task_times
        else:
            return result


class FeatProcessEmbed(nn.Module):
    def __init__(self, recom_host: ModuleType, device: torch.device, embed_tables: List[torch.Tensor] = None,
                 table_shapes: List[Tuple[int, int]] = None, librecom_path: str = None):
        super(FeatProcessEmbed, self).__init__()
        self.device = device

        if librecom_path:
            torch.ops.load_library(librecom_path)

        self.preprocess_layer = FeatPreprocess(recom_host, device=self.device)
        self.embed_layer = FeatEmbed(device=self.device, embed_tables=embed_tables, table_shapes=table_shapes)
        self.max_concurrent_blocks = self.embed_layer.max_concurrent_blocks

    def forward(self, raw_data: List[List[str]], timing: bool = False, timing_schedules: int = None,
                timing_kernel: bool = False):
        preprocess_result = self.preprocess_layer(raw_data, self.max_concurrent_blocks)
        if not timing_kernel:
            result = self.embed_layer(*preprocess_result, timing=timing, timing_schedules=timing_schedules)
            return result
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = self.embed_layer(*preprocess_result, timing=timing, timing_schedules=timing_schedules)
            end.record()
            end.synchronize()
            return result, start.elapsed_time(end)


class RecomModel(nn.Module):
    def __init__(self, mlp_unit_nums: List[int], device: torch.device, embed_tables: List[torch.Tensor] = None,
                 table_shapes: List[Tuple[int, int]] = None):
        super(RecomModel, self).__init__()
        self.device = device
        self.embed_layer = FeatEmbed(device=self.device, embed_tables=embed_tables, table_shapes=table_shapes)

        layers = nn.ModuleList()
        unit_nums = [self.embed_layer.output_units] + list(mlp_unit_nums) + [1]
        for i in range(0, len(unit_nums) - 1):
            in_units = unit_nums[i]
            out_units = unit_nums[i + 1]
            layers.append(nn.Linear(in_units, out_units, bias=True, device=self.device))
            if i + 1 == len(unit_nums) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, arg_buffers: torch.Tensor, data_buffer_offsets: torch.Tensor,
                scala_args: torch.Tensor, extra_buffers: torch.Tensor, extra_buffer_offsets: torch.Tensor,
                task_mapping: torch.Tensor, task_tiles: torch.Tensor, task_barriers: torch.Tensor,
                embed_output_size: int):
        embed_output: torch.Tensor = self.embed_layer(
            arg_buffers,
            data_buffer_offsets,
            scala_args,
            extra_buffers,
            extra_buffer_offsets,
            task_mapping,
            task_tiles,
            task_barriers,
            embed_output_size,
        )
        embed_output = torch.reshape(embed_output, [-1, self.embed_layer.output_units])
        output = self.mlp(embed_output)
        return output
