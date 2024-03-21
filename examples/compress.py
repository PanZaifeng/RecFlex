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
import shutil
from typing import Tuple, List, Dict

from RecFlex.parser import parse_table_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def compress_model(model_dir: str, compressed_model_dir: str) -> None:
    table_shapes: List[Tuple[int, int]] = parse_table_shapes(os.path.join(model_dir, "table_config.txt"))
    with open(os.path.join(model_dir, "data_config.txt")) as orig_data_config:
        with open(os.path.join(compressed_model_dir, "data_config.txt"), "w") as compressed_data_config:
            feat_id = 0
            cluster_mappings: List[Dict[Tuple[int, int], List[int]]] = []
            for line in orig_data_config.readlines():
                feat_config = line.split(",")
                feat_type, feat_gen_num = feat_config[:2]
                cluster_mapping: Dict[Tuple[int, int], List[int]] = {}
                for i in range(int(feat_gen_num)):
                    table_shape = tuple(table_shapes[feat_id])
                    if table_shape in cluster_mapping.keys():
                        cluster_mapping[table_shape].append(feat_id)
                    else:
                        cluster_mapping[table_shape] = [feat_id]
                    feat_id += 1
                cluster_mappings.append(cluster_mapping)

                for cluster_feats in cluster_mapping.values():
                    compressed_data_config.write(','.join([feat_type, "1"] + feat_config[2:]))

    with open(os.path.join(compressed_model_dir, "feat_map.txt"), "w") as feat_map:
        for cluster_mapping in cluster_mappings:
            for cluster_feats in cluster_mapping.values():
                feat_map.write(",".join(map(str, cluster_feats)) + "\n")

    with open(os.path.join(compressed_model_dir, "table_config.txt"), "w") as compressed_table_config:
        for cluster_mapping in cluster_mappings:
            for table_shape in cluster_mapping.keys():
                compressed_table_config.write(",".join(map(str, table_shape)) + "\n")

    with open(os.path.join(model_dir, "schedule_config.txt")) as schedule_config:
        schedule_config_lines = schedule_config.readlines()

    with open(os.path.join(compressed_model_dir, "schedule_config.txt"), "w") as compressed_schedule_config:
        for cluster_mapping in cluster_mappings:
            for cluster_feats in cluster_mapping.values():
                compressed_schedule_config.write(schedule_config_lines[cluster_feats[0]])

    data_dir = os.path.join(model_dir, "data")
    compressed_data_dir = os.path.join(compressed_model_dir, "data")
    os.makedirs(compressed_data_dir, exist_ok=True)
    shutil.copy(src=os.path.join(data_dir, "batch_sizes.txt"),
                dst=os.path.join(compressed_data_dir, "batch_sizes.txt"))

    compressed_feat_id = 0
    for cluster_mapping in cluster_mappings:
        for cluster_feats in cluster_mapping.values():
            orig_feat_id = cluster_feats[0]
            shutil.copy(src=os.path.join(data_dir, f"f{orig_feat_id}.txt"),
                        dst=os.path.join(compressed_data_dir, f"f{compressed_feat_id}.txt"))
            compressed_feat_id += 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    compress_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
