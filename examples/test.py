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
import time
import torch
import torch.nn.functional as F
from typing import List, Tuple, NamedTuple
from types import ModuleType

from RecFlex.code_emitter import ScheduleBase
from RecFlex.rec_modules import FeatProcessEmbed
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes
from RecFlex.utils import import_from_path, find_file, compile_build
from RecFlex.cuda_helper import CudaProfiler
from RecFlex.tuning import Tuner, TuneConfig, BasicConfig

from schedules.reduce_by_key import ReduceByKeySchedule, ReduceByKeyParams
from schedules.multi_block_per_sample import MultiBlockPerSampleSchedule, MultiBlockPerSampleParams
from schedules.one_block_multi_sample import OneBlockMultiSampleSchedule, OneBlockMultiSampleParams
from schedules.warp_per_sample import WarpPerSampleSchedule, WarpPerSampleParams
from schedules.one_hot import OneHotSchedule, OneHotParams
from schedules.one_hot_multi_block import OneHotMultiBlockSchedule, OneHotMultiBlockParams

from tune import create_tuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-n", "--num_processes", type=int, default=64)
    parser.add_argument("-b", "--block_threads", type=int, default=256)
    parser.add_argument("-g", "--occupancy", type=int)
    parser.add_argument("--sample_batch_nums", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompile", action="store_true", default=False)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def build_targets(args: argparse.Namespace, feat_schedules: List[Tuple[ScheduleBase, NamedTuple]],
                  tuner: Tuner) -> str:
    build_dir = os.path.join(args.output_dir, "build")
    if not os.path.exists(f"{build_dir}/librecom.so") or args.recompile:
        tuner.emit_codes(args.output_dir, feat_schedules, args.occupancy, debug=args.debug)
        compile_build(build_dir)
    return build_dir


def create_model(table_shapes: List[int], recom_host: ModuleType):
    cuda_dev = torch.cuda.current_device()
    model = FeatProcessEmbed(recom_host, cuda_dev, table_shapes=table_shapes)
    return model


def check_results(results: torch.Tensor, h_embed_tables: List[torch.Tensor],
                  raw_data: List[List[str]]):
    dim_sum = 0
    for table in h_embed_tables:
        dim_sum += table.shape[1]
    results = torch.reshape(results, [-1, dim_sum]).cpu()

    dim_offset = 0
    for raw, table in zip(raw_data, h_embed_tables):
        next_dim_offset = dim_offset + table.shape[1]
        for sid, row in enumerate(raw):
            if row == "":
                indices = []
            else:
                indices = list(map(int, row.strip().split(',')))
            lookup = F.embedding(torch.LongTensor(indices), table)
            pooling = lookup.sum(0)
            assert torch.allclose(pooling, results[sid, dim_offset:next_dim_offset], atol=1e-4, rtol=1e-4)
        dim_offset = next_dim_offset


def measure(model: FeatProcessEmbed, raw_data_batches: List[List[List[str]]], check: bool = False) -> int:
    if check:
        h_embed_tables = [table.cpu() for table in model.embed_layer.embed_tables]
    cu_prof = CudaProfiler()
    cu_prof.start()
    accum_time_ms = 0
    for raw_data in raw_data_batches:
        start = time.perf_counter()
        output = model(raw_data)
        end = time.perf_counter()
        if check:
            check_results(output, h_embed_tables, raw_data)
        accum_time_ms += (end - start) * 1000
    cu_prof.stop()
    return accum_time_ms


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tuner = create_tuner(args)
    metas = tuner.metas

    # feat_schedules: List[Tuple[ScheduleBase, NamedTuple]] = [
    #     (OneBlockMultiSampleSchedule(meta_data=metas[i], block_threads=tuner.basic.block_threads),
    #      OneBlockMultiSampleParams(vector_size=4, items_per_thread=4, vblock_dim_x=16, vblock_dim_y=4))
    #     for i in range(1000)
    # ]
    feat_schedules = []
    build_dir = build_targets(args, feat_schedules, tuner)

    librecom_path = f"{build_dir}/librecom.so"
    recom_host_path = find_file(build_dir, "recom_host*.so")
    recom_host = import_from_path("recom_host", recom_host_path)
    torch.ops.load_library(librecom_path)

    table_shapes = parse_table_shapes(tuner.basic.table_config_fname)
    raw_data_batches = parse_raw_data_batches(tuner.basic.batch_sizes_fname, tuner.basic.data_fnames,
                                              args.sample_batch_nums, seed=args.seed, feat_major=False)

    model = create_model(table_shapes, recom_host)
    t = measure(model, raw_data_batches, args.check)
    print(f"Accumulated end-to-end time (ms): {t}")


if __name__ == "__main__":
    main()
