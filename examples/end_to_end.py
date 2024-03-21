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
from typing import List
from types import ModuleType

from RecFlex.rec_modules import FeatPreprocess, RecomModel
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes
from RecFlex.utils import import_from_path, find_file
from RecFlex.cuda_helper import CudaProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--mlp_unit_nums", type=int, nargs="+", default=[1024, 256, 128])
    parser.add_argument("-t", "--table_config_fname", type=str, required=True)
    parser.add_argument("-b", "--build_dir", type=str, required=True)
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--sample_batch_nums", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args

def create_models(mlp_unit_nums: List[int], table_shapes: List[int], recom_host: ModuleType):
    cuda_dev = torch.cuda.current_device()
    preprocess = FeatPreprocess(recom_host, cuda_dev)
    recom_model = RecomModel(mlp_unit_nums, cuda_dev, table_shapes=table_shapes)
    return preprocess, recom_model

def measure(preprocess: FeatPreprocess, recom_model: RecomModel, max_concurrent_blocks: int,
            raw_data_batches: List[List[List[str]]]) -> int:
    accum_time_ms = 0
    for raw_data in raw_data_batches:
        preprocess_result = preprocess(raw_data, max_concurrent_blocks)
        start = time.perf_counter()
        output = recom_model(*preprocess_result)
        end = time.perf_counter()
        accum_time_ms += (end - start) * 1000
    return accum_time_ms

def main():
    args = parse_args()

    librecom_path = f"{args.build_dir}/librecom.so"
    recom_host_path = find_file(args.build_dir, "recom_host*.so")
    recom_host = import_from_path("recom_host", recom_host_path)
    torch.ops.load_library(librecom_path)

    table_shapes = parse_table_shapes(args.table_config_fname)
    data_fnames = [f"{args.data_dir}/f{feat_id}.txt" for feat_id in range(len(table_shapes))]
    batch_sizes_fname = f"{args.data_dir}/batch_sizes.txt"
    raw_data_batches = parse_raw_data_batches(batch_sizes_fname, data_fnames, args.sample_batch_nums,
                                              seed=args.seed, feat_major=False)

    preprocess, recom_model = create_models(args.mlp_unit_nums, table_shapes, recom_host)
    max_concurrent_blocks = torch.ops.recom.get_max_concurrent_blocks()

    cu_prof = CudaProfiler()
    cu_prof.start()
    t = measure(preprocess, recom_model, max_concurrent_blocks, raw_data_batches)
    cu_prof.stop()

    print(f"Accumulated end-to-end time (ms): {t}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(t)


if __name__ == "__main__":
    main()