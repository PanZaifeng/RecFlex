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
import numpy as np
from typing import List, Tuple, Dict, Any

from RecFlex.cuda_helper import CudaProfiler
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

tf.disable_resource_variables()
tf.disable_eager_execution()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-c", "--data_config_path", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("-l", "--lib_path", type=str)
    parser.add_argument("--mlp_unit_nums", type=int, nargs="+", default=[1024, 256, 128])
    args = parser.parse_args()
    return args


def parse_feat_types(data_config_path: str) -> List[bool]:
    onehot_flags: List[bool] = []
    with open(data_config_path) as config_file:
        for line in config_file.readlines():
            feat_config = line.split(",")
            feat_type, feat_gen_num = feat_config[:2]
            feat_config = list(
                map(lambda x: int(x) if "." not in x else float(x), feat_config[2:])
            )

            feat_gen_num = int(feat_gen_num)
            if feat_type == "one-hot":
                onehot_flags.extend([True] * feat_gen_num)
            elif feat_type in ["multi-hot", "multi-hot-static", "multi-hot-one-side"]:
                onehot_flags.extend([False] * feat_gen_num)
            else:
                raise Exception(f"Invalid feature type {feat_type}")

    return onehot_flags


def parse_sparse_data(data_dir: str, row_nums: List[int], onehot_flags: List[bool]) -> List[Dict[str, Any]]:
    num_feats = len(row_nums)
    raw_data_batches = parse_raw_data_batches(batch_sizes_fname=f"{data_dir}/batch_sizes.txt",
                                              data_fnames=[f"{data_dir}/f{feat_id}.txt" for feat_id in range(num_feats)],
                                              feat_major=False)

    feed_dicts: List[Dict[str, Any]] = []
    for raw_data_batch in raw_data_batches:
        feed_dict = {}
        for i, (raw_feat_batch, is_onehot) in enumerate(zip(raw_data_batch, onehot_flags)):
            if is_onehot:
                feed_dict[f"f{i}:0"] = np.array(list(map(int, raw_feat_batch)), dtype=np.int32)
            else:
                indices = []
                values = []
                for segid, raw in enumerate(raw_feat_batch):
                    raw = raw.strip()
                    if raw != "":
                        value = list(map(int, raw.split(",")))
                    else:
                        value = []
                    indices.extend([segid] * len(value))
                    values.extend(value)
                feed_dict[f"f{i}_indices:0"] = indices
                feed_dict[f"f{i}_values:0"] = values
                feed_dict[f"f{i}_shape:0"] = len(raw_feat_batch)
        feed_dicts.append(feed_dict)

    return feed_dicts


def create_embed_layer(table_shapes: List[Tuple[int, int]], onehot_flags: List[bool]):
    feat_outputs = []
    for i, ((rows, embed_dim), is_onehot) in enumerate(zip(table_shapes, onehot_flags)):
        embed_table = tf.Variable(tf.random_normal([rows, embed_dim]), name=f"table{i}")
        if is_onehot:
            placeholder = tf.placeholder(dtype=tf.int32, name=f"f{i}", shape=[None])
            feat_output = tf.gather(embed_table, placeholder)
        else:
            sp_indices = tf.placeholder(dtype=tf.int32, name=f"f{i}_indices", shape=[None])
            sp_values = tf.placeholder(dtype=tf.int32, name=f"f{i}_values", shape=[None])
            batch_size = tf.placeholder(dtype=tf.int32, name=f"f{i}_shape", shape=())
            feat_output = tf.sparse.segment_sum(embed_table, sp_values, sp_indices, num_segments=batch_size)
        feat_outputs.append(feat_output)

    return tf.concat(feat_outputs, axis=1)


def create_mlp(input, mlp_unit_nums: List[int]):
    x = input
    for i, hidden_units in enumerate(mlp_unit_nums):
        x = tf.layers.dense(x, units=hidden_units,
                            activation=tf.nn.relu,
                            name=f'mlp{i}')
    logits = tf.layers.dense(x, units=1, activation=None)
    probability = tf.math.sigmoid(logits)
    return probability


def main():
    args = parse_args()

    table_shapes = parse_table_shapes(args.table_config_path)
    row_nums = [row_num for row_num, embed_dim in table_shapes]
    onehot_flags = parse_feat_types(args.data_config_path)

    embed = create_embed_layer(table_shapes, onehot_flags)
    output = create_mlp(embed, args.mlp_unit_nums)
    feed_dicts = parse_sparse_data(args.data_dir, row_nums, onehot_flags)

    if args.lib_path:
        tf.load_op_library(args.lib_path)
    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0,
                            device_count={"GPU": 1})
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        cu_prof = CudaProfiler()
        cu_prof.start()
        for feed_dict in feed_dicts:
            sess.run(output, feed_dict=feed_dict)
        cu_prof.stop()


if __name__ == "__main__":
    main()
