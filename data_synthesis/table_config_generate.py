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
import random


ONEHOT_CHOICES = [ 4, 8, 16, 32, 64, 128 ]
ONEHOT_WEIGHTS = [ 8, 8, 4, 2, 1, 1 ]
ONEHOT_ROWS = 1024

MULTIHOT_CHOICES = [ 4, 8, 16, 32, 64, 128 ]
MULTIHOT_WEIGHTS = [ 2, 2, 2, 1, 1, 1 ]
MULTIHOT_ROWS = 4096
MULTIHOT_ONE_SIDE_ROWS = 4096 * 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--data_config_path", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("-d", "--fixed_embed_dim", type=int)
    args = parser.parse_args()

    row_nums = []
    embed_dims = []
    with open(args.data_config_path) as config_file:
        for line in config_file.readlines():
            feat_config = line.split(",")
            feat_type, feat_gen_num = feat_config[:2]
            feat_config = list(
                map(lambda x: int(x) if "." not in x else float(x), feat_config[2:])
            )

            feat_gen_num = int(feat_gen_num)
            if feat_type == "one-hot":
                row_nums.extend([ONEHOT_ROWS] * feat_gen_num)
                embed_dims.extend(random.choices(ONEHOT_CHOICES, ONEHOT_WEIGHTS, k=feat_gen_num))
            elif feat_type in ["multi-hot", "multi-hot-static"]:
                row_nums.extend([MULTIHOT_ROWS] * feat_gen_num)
                embed_dims.extend(random.choices(MULTIHOT_CHOICES, MULTIHOT_WEIGHTS, k=feat_gen_num))
            elif feat_type == "multi-hot-one-side":
                row_nums.extend([MULTIHOT_ONE_SIDE_ROWS] * feat_gen_num)
                embed_dims.extend(random.choices(MULTIHOT_CHOICES, MULTIHOT_WEIGHTS, k=feat_gen_num))

    if args.fixed_embed_dim:
        embed_dims = [args.fixed_embed_dim for _ in range(len(row_nums))]

    with open(args.table_config_path, "w") as table_config:
        for row_num, embed_dim in zip(row_nums, embed_dims):
            table_config.write(f"{row_num},{embed_dim}\n")
