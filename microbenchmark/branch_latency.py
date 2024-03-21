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

import argparse
import os
import multiprocessing as mp
import numpy as np

from RecFlex.code_emitter import CodeEmitter, MetaData
from RecFlex.utils import compile_build

from sleep_schedule import SleepSchedule, SleepParams
from fence_schedule import FenceSchedule, FenceParams


def emit_codes(num_feats: int, code_fname: str, cmake_fname: str, cmake_extra: str, inline: bool, occupancy: int, empty: bool):
    if empty:
        emitter = CodeEmitter([[FenceSchedule(meta_data=MetaData(32), block_threads=256), FenceParams(ident=i)]
                                for i in range(num_feats)])
    else:
        emitter = CodeEmitter([[SleepSchedule(meta_data=MetaData(32), block_threads=256), SleepParams(ident=i)]
                                for i in range(num_feats)])
    emitter.emit(ofname=code_fname, inline=inline, occupancy=occupancy, timing=True, tune=True, timing_branch=True)
    emitter.emit_cmake_list(ofname=cmake_fname, code_fname=os.path.basename(code_fname), host_fname=None, extra=cmake_extra)

def measure(librecom_path: str, inline: bool, num_feats: int, blocks_per_feat: int, sleep_time_in_us: int,
            result_queue: mp.Queue):
    import torch
    torch.ops.load_library(librecom_path)
    cuda_dev = torch.cuda.current_device()
    embed_table = torch.ones(size=[4096, 32], dtype=torch.float32, device=cuda_dev)
    embed_tables = [embed_table for _ in range(num_feats)]
    embed_table_ptrs = torch.ops.recom.get_gpu_pointers_array(embed_tables)
    
    arg_buffers = torch.zeros([0], dtype=torch.int32, device=cuda_dev)
    data_buffer_offsets = torch.zeros([num_feats + 1], dtype=torch.int32, device=cuda_dev)
    extra_buffers = torch.zeros([0], dtype=torch.int8, device=cuda_dev)
    extra_buffer_offsets = torch.zeros([num_feats + 1], dtype=torch.int32, device=cuda_dev)
    task_barriers = torch.zeros([0], dtype=torch.int32, device=cuda_dev)

    scala_args = torch.ones(num_feats, dtype=torch.int32, device=cuda_dev) * sleep_time_in_us
    output_offsets = torch.arange(start=0, end=num_feats + 1, step=blocks_per_feat * 4,
                                  dtype=torch.int32, device=cuda_dev)
    output_size = num_feats * blocks_per_feat * 4
    max_concurrent_blocks = blocks_per_feat

    task_mapping = torch.tensor([[i, j] for i in range(num_feats) for j in range(blocks_per_feat)],
                                dtype=torch.int32, device=cuda_dev)
    task_tiles = torch.ones([num_feats], dtype=torch.int32, device=cuda_dev) * blocks_per_feat

    output, times = torch.ops.recom.process(
        embed_table_ptrs,
        arg_buffers,
        data_buffer_offsets,
        scala_args,
        output_offsets,
        extra_buffers,
        extra_buffer_offsets,
        task_mapping,
        task_tiles,
        task_barriers,
        output_size,
        max_concurrent_blocks
    )

    tmean = torch.mean(times).cpu().item()
    tmin = torch.min(times).cpu().item()
    tmax = torch.max(times).cpu().item()

    result_queue.put([inline, tmean, tmin, tmax])


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_feats", type=int, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-b", "--blocks_per_feat", type=int, required=1)
    parser.add_argument("-t", "--sleep_time", type=int, default=1000)
    parser.add_argument("-g", "--occupancy", type=int, default=1)
    parser.add_argument("-e", "--empty", action="store_true")
    args = parser.parse_args()

    cmake_extra = '''
target_compile_options(recom PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                        --ptxas-options=-v;
                        --keep;
                       >)
'''

    for inline in [True, False]:
        work_dir = f"{args.output_dir}/inline_{inline}"
        build_dir = f"{work_dir}/build"
        if not os.path.exists(f"{build_dir}/librecom.so"):
            os.makedirs(build_dir, exist_ok=True)
            emit_codes(num_feats=args.num_feats, inline=inline,
                       code_fname=f"{work_dir}/process.cu",
                       occupancy=args.occupancy, empty=args.empty,
                       cmake_fname=f"{work_dir}/CMakeLists.txt",
                       cmake_extra=cmake_extra)
            compile_build(build_dir)

    result_queue = mp.Queue()
    for inline in [True, False]:
        work_dir = f"{args.output_dir}/inline_{inline}"
        build_dir = f"{work_dir}/build"
        p = mp.Process(target=measure, args=(f"{build_dir}/librecom.so", inline, args.num_feats,
                                             args.blocks_per_feat, args.sleep_time, result_queue))
        p.start()
        p.join()

    while not result_queue.empty():
        inline, tmean, tmin, tmax = result_queue.get()
        print([inline], tmean, tmin, tmax)