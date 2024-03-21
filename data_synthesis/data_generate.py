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
import sys
import random
import time
import math
import multiprocessing as mp
import numpy as np
from scipy.stats import truncnorm

from RecFlex import TaskManager
from RecFlex.parser import parse_table_shapes
from RecFlex.utils import os_check


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("-n", "--num_processes", type=int, default=64)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--query_mean", type=int, default=256)
    parser.add_argument("--query_stddev", type=int, default=128)
    parser.add_argument("--query_max", type=int, default=512)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def get_power_law_probabilities(size, alpha=3.0):
    p = np.random.power(a=alpha, size=size) 
    return p / np.sum(p)


def lognomral_generator(mean, stddev, size=1, dtype=np.int32):
    normal_random = np.random.normal(math.log(mean), math.log(stddev), size)
    lognormal_random = np.exp(normal_random)
    lognormal_random = lognormal_random.astype(dtype)

    return lognormal_random


def truncated_nomral_generator(mean, stddev, lb, ub, size=1, dtype=np.int32):
    lower, upper = (lb - mean) / stddev, (ub - mean) / stddev
    normal_random = truncnorm.rvs(lower, upper, loc=mean, scale=stddev, size=size)
    normal_random = np.array(normal_random, dtype=dtype)

    return normal_random


def one_side_truncated_nomral_generator(mean, stddev, size=1, dtype=np.int32):
    return truncated_nomral_generator(mean, stddev, 0, np.inf, size, dtype)


def batch_size_generator(mean, stddev, maximum, size=1):
    batch_sizes = lognomral_generator(mean, stddev, size)

    def mapf(bs):
        if bs <= maximum:
            return max(1, bs)
        else:
            splits = (bs + maximum - 1) // maximum
            if random.randint(1, splits) == splits:
                return bs % maximum
            else:
                return maximum

    return np.array(list(map(mapf, batch_sizes)))


def parse_data_config(config_path):
    feat_types = []
    feat_gen_nums = []
    feat_configs = []
    with open(config_path) as config_file:
        for line in config_file.readlines():
            feat_config = line.split(",")
            feat_type, feat_gen_num = feat_config[:2]
            feat_config = list(
                map(lambda x: int(x) if "." not in x else float(x), feat_config[2:])
            )

            feat_types.append(feat_type)
            feat_gen_nums.append(int(feat_gen_num))
            feat_configs.append(feat_config)
    return feat_types, feat_gen_nums, feat_configs


def generate_batch_sizes(args):
    fname = f"{args.output_dir}/batch_sizes.txt"
    if os.path.exists(fname):
        batch_sizes = []
        with open(fname) as f:
            for line in f.readlines():
                batch_sizes.append(int(line.strip()))
    else:
        np.random.seed(int(time.time()))
        batch_sizes = batch_size_generator(args.query_mean, args.query_stddev, args.query_max,
                                        args.num_queries)
        temp = fname + ".tmp"
        with open(temp, "w") as f:
            for bs in batch_sizes:
                f.write(f"{bs}\n")
        os_check(f"mv {temp} {fname}")
    return batch_sizes


def generate_data_task(fname, hot_counts_list, absents_list, index_candidates,
                       power_law_probabilities):
    print(f"Generating {fname}")
    np.random.seed(int(time.time()) + os.getpid())
    temp = fname + ".tmp"
    with open(temp, "w") as ofile:
        for hot_counts, absents in zip(
            hot_counts_list, absents_list
        ):
            for hot_count, absent in zip(hot_counts, absents):
                s = ""
                if not absent:
                    indices = np.random.choice(
                        index_candidates, hot_count, replace=True, p=power_law_probabilities
                    )
                    s = ",".join(map(str, indices))
                ofile.write(s + "\n")
    os_check(f"mv {temp} {fname}")
    print(f"Finished {fname}")


def generate_data(feat_types, feat_gen_nums, feat_configs, batch_sizes, row_nums,
                  num_processes, output_dir):
    task_manager = TaskManager(list(range(num_processes)))
    feat_id = 0
    for feat_type, feat_gen_num, feat_config in zip(
        feat_types, feat_gen_nums, feat_configs
    ):
        if feat_type == "one-hot":
            (coverage,) = feat_config
            hot_counts_list = [np.ones(bs, dtype=np.int32) for bs in batch_sizes]
        elif feat_type == "multi-hot-static":
            hot_count, coverage = feat_config
            hot_counts_list = [np.ones(bs, dtype=np.int32) * hot_count for bs in batch_sizes]
        elif feat_type == "multi-hot":
            mean, stddev, maximum, coverage = feat_config
            hot_counts_list = [
                truncated_nomral_generator(mean, stddev, 0, maximum, bs)
                for bs in batch_sizes
            ]
        elif feat_type == "multi-hot-one-side":
            mean, stddev, coverage = feat_config
            hot_counts_list = [
                one_side_truncated_nomral_generator(mean, stddev, bs) for bs in batch_sizes
            ]

        absents_list = [np.random.random(bs) > coverage for bs in batch_sizes]
        power_law_probabilities = get_power_law_probabilities(size=row_nums[feat_id])
        index_candidates = np.arange(0, row_nums[feat_id])

        for i in range(feat_gen_num):
            fname = f"{output_dir}/f{feat_id}.txt"
            if not os.path.exists(fname):
                task_manager.add_task(target=generate_data_task, args=(fname, hot_counts_list, absents_list,
                                                                       index_candidates, power_law_probabilities))
            feat_id += 1
    task_manager.join()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    feat_types, feat_gen_nums, feat_configs = parse_data_config(args.config_path)
    table_shapes = parse_table_shapes(args.table_config_path)
    row_nums = [row_num for row_num, embed_dim in table_shapes]
    batch_sizes = generate_batch_sizes(args)
    generate_data(feat_types, feat_gen_nums, feat_configs, batch_sizes, row_nums,
                  args.num_processes, args.output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
