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
import time
import torch
import torch.nn as nn

from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec import EmbeddingBagConfig, KeyedJaggedTensor, JaggedTensor
from RecFlex.cuda_helper import CudaProfiler
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes

from typing import List, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("-l", "--mlp_unit_nums", type=int, nargs="+", default=[1024, 256, 128])
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    return args


def parse_sparse_data(data_dir: str, num_feats: int) -> List[KeyedJaggedTensor]:
    raw_data_batches = parse_raw_data_batches(batch_sizes_fname=f"{data_dir}/batch_sizes.txt",
                                              data_fnames=[f"{data_dir}/f{feat_id}.txt" for feat_id in range(num_feats)],
                                              feat_major=False)

    kjt_list = []
    for raw_data_batch in raw_data_batches:
        jt_dict: Dict[str, JaggedTensor] = {}
        for feat_id, feat_batch in enumerate(raw_data_batch):
            values: List[torch.Tensor] = []
            for raw in feat_batch:
                raw = raw.strip()
                if raw != "":
                    value = torch.tensor(list(map(int, raw.split(','))), dtype=torch.int32)
                else:
                    value = torch.tensor([], dtype=torch.int32)
                values.append(value)
            jt = JaggedTensor.from_dense(values=values)
            jt_dict[f"f{feat_id}"] = jt
        kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)
        kjt_list.append(kjt)
    return kjt_list


def make_fuse_ebc(table_shapes: List[Tuple[int, int]], device: torch.device) -> FusedEmbeddingBagCollection:
    embedding_bag_configs: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            name=f"ebc_{feat_id}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"f{feat_id}"],
        )
        for feat_id, (num_embeddings, embedding_dim) in enumerate(table_shapes)
    ]

    fused_ebc = FusedEmbeddingBagCollection(tables=embedding_bag_configs,
                                            optimizer_type=torch.optim.SGD,
                                            optimizer_kwargs={"lr": 0.02},
                                            device=device)
    return fused_ebc


def make_mlp(input_units: int, mlp_unit_nums: List[int], device: torch.device) -> nn.Sequential:
    layers = nn.ModuleList()
    unit_nums = [input_units] + list(mlp_unit_nums) + [1]
    for i in range(0, len(unit_nums) - 1):
        in_units = unit_nums[i]
        out_units = unit_nums[i + 1]
        layers.append(nn.Linear(in_units, out_units, bias=True, device=device))
        if i + 1 == len(unit_nums) - 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers)
    return mlp


def main():
    args = parse_args()

    table_shapes = parse_table_shapes(args.table_config_path)
    embed_output_units = sum([embed_dim for row_num, embed_dim in table_shapes])
    device = torch.device("cuda")
    fused_ebc = make_fuse_ebc(table_shapes, device)
    mlp = make_mlp(embed_output_units, args.mlp_unit_nums, device)
    kjt_list = parse_sparse_data(args.data_dir, len(table_shapes))

    cu_prof = CudaProfiler()
    cu_prof.start()
    accum_time = 0
    for kjt in kjt_list:
        kjt = kjt.to(device)
        t1 = time.time()
        # TODO: write an nn.Module
        sparse_features = fused_ebc(kjt).values()
        out = mlp(sparse_features)
        t2 = time.time()
        accum_time += t2 - t1
    cu_prof.stop()

    if args.output:
        with open(args.output, "w") as f:
            f.write(str(accum_time * 1000))


if __name__ == "__main__":
    main()
