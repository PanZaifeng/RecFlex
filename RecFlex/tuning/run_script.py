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
import torch

from RecFlex.rec_modules import FeatProcessEmbed
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes
from RecFlex.utils import import_from_path, find_file
from RecFlex.cuda_helper import CudaProfiler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_config_fname", type=str, required=True)
    parser.add_argument("--build_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--spec_feat_id", type=int)
    parser.add_argument("--sample_batch_nums", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    librecom_path = f"{args.build_dir}/librecom.so"
    recom_host_path = find_file(args.build_dir, "recom_host*.so")
    recom_host = import_from_path("recom_host", recom_host_path)
    torch.ops.load_library(librecom_path)

    table_shapes = parse_table_shapes(args.table_config_fname)
    if args.spec_feat_id:
        table_shapes = [table_shapes[args.spec_feat_id]]
    cuda_dev = torch.cuda.current_device()
    model = FeatProcessEmbed(recom_host=recom_host, table_shapes=table_shapes, device=cuda_dev)

    if args.spec_feat_id:
        data_fnames = [f"{args.data_dir}/f{args.spec_feat_id}.txt"]
    else:
        data_fnames = [f"{args.data_dir}/f{feat_id}.txt" for feat_id in range(len(table_shapes))]
    batch_sizes_fname = f"{args.data_dir}/batch_sizes.txt"
    raw_data_batches = parse_raw_data_batches(batch_sizes_fname, data_fnames, args.sample_batch_nums,
                                              seed=args.seed, feat_major=False)

    cu_prof = CudaProfiler()
    cu_prof.start()
    for raw_data in raw_data_batches:
        output = model(raw_data)
    torch.cuda.synchronize()
    cu_prof.stop()