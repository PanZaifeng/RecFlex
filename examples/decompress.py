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
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model_dir", type=str, required=True)
    parser.add_argument("-o", "--output_model_dir", type=str, required=True)
    parser.add_argument("--naive", action="store_true", default=False)
    args = parser.parse_args()
    return args


def parse_clusters(feat_map_path: str) -> List[List[int]]:
    clusters: List[List[int]] = []
    with open(feat_map_path) as feat_map:
        for line in feat_map.readlines():
            cluster_feats = list(map(int, line.strip().split(',')))
            clusters.append(cluster_feats)
    return clusters


def restore_output(input_codegen_dir: str, output_codegen_dir: str, clusters: List[List[int]]) -> None:
    os.makedirs(output_codegen_dir, exist_ok=True)
    for feat_id, cluster_feats in enumerate(clusters):
        for output_feat in cluster_feats:
            shutil.copytree(src=os.path.join(input_codegen_dir, f"f{feat_id}"),
                            dst=os.path.join(output_codegen_dir, f"f{output_feat}"),
                            dirs_exist_ok=True)


def main():
    args = parse_args()
    clusters = parse_clusters(os.path.join(args.input_model_dir, "feat_map.txt"))
    if not args.naive:
        restore_output(os.path.join(args.input_model_dir, "output"),
                       os.path.join(args.output_model_dir, "output"), clusters)
    else:
        input_naive = os.path.join(args.input_model_dir, "output", "naive")
        if os.path.exists(input_naive):
            restore_output(input_naive, os.path.join(args.output_model_dir, "output", "naive"), clusters)


if __name__ == "__main__":
    main()
