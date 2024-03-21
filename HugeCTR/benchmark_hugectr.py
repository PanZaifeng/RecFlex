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

import hugectr
from hugectr.tools import DataGeneratorParams, DataGenerator
from hugectr.inference import InferenceParams, CreateInferenceSession

import argparse
import os
import time
import numpy as np
from typing import List, Tuple

from RecFlex.cuda_helper import CudaProfiler
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-c", "--data_config_path", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("-g", "--gpu_cache", action="store_true", default=False)
    parser.add_argument("-l", "--mlp_unit_nums", type=int, nargs="+", default=[1024, 256, 128])
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    return args


def parse_slot_nnz_array(data_config_path: str) -> List[int]:
    slot_nnz_array: List[int] = []
    with open(data_config_path) as config_file:
        for line in config_file.readlines():
            feat_config = line.split(",")
            feat_type, feat_gen_num = feat_config[:2]
            feat_gen_num = int(feat_gen_num)
            feat_config = list(
                map(lambda x: int(x) if "." not in x else float(x), feat_config[2:])
            )

            if feat_type == "one-hot":
                slot_nnz_array.extend([1] * feat_gen_num)
            elif feat_type == "multi-hot-static":
                hot_count, coverage = feat_config
                slot_nnz_array.extend([hot_count] * feat_gen_num)
            elif feat_type == "multi-hot":
                mean, stddev, maximum, coverage = feat_config
                slot_nnz_array.extend([maximum] * feat_gen_num)
            else:
                raise Exception(f"Unsupported type {feat_type}")

    return slot_nnz_array


def parse_sparse_data(data_dir: str, row_nums: List[int]) -> List[Tuple[List[int], List[int]]]:
    num_feats = len(row_nums)
    raw_data_batches = parse_raw_data_batches(batch_sizes_fname=f"{data_dir}/batch_sizes.txt",
                                              data_fnames=[f"{data_dir}/f{feat_id}.txt" for feat_id in range(num_feats)],
                                              feat_major=False)

    shifts = np.insert(np.cumsum(row_nums), 0, 0)
    batches: List[Tuple[List[int], List[int]]] = []
    for raw_data_batch in raw_data_batches:
        batch_size = len(raw_data_batch[0])
        embedding_columns: List[int] = []
        row_ptrs: List[int] = [0]
        for sample_id in range(batch_size):
            for feat_id in range(num_feats):
                raw = raw_data_batch[feat_id][sample_id].strip()
                if raw != "":
                    value = [int(x) + shifts[feat_id] for x in raw.split(',')]
                else:
                    value = []
                embedding_columns.extend(value)
                row_ptrs.append(row_ptrs[-1] + len(value))
        batches.append([embedding_columns, row_ptrs])
    return batches


def generate_train_data(row_nums: List[int], source: str, eval_source: str):
    data_generator_params = DataGeneratorParams(
        format=hugectr.DataReaderType_t.Raw,
        label_dim=1,
        dense_dim=0,
        num_slot=len(row_nums),
        i64_input_key=False,
        source=source,
        eval_source=eval_source,
        slot_size_array=row_nums,
        check_type=hugectr.Check_t.Sum,
        dist_type=hugectr.Distribution_t.PowerLaw,
        power_law_type=hugectr.PowerLaw_t.Short,
        num_files=1,
        eval_num_files=1,
        num_samples=256,
        eval_num_samples=256,
        float_label_dense=False,
    )
    data_generator = DataGenerator(data_generator_params)
    data_generator.generate()


def build_model(model_name: str, model_config_path: str, row_nums: List[int], embed_dim: int,
                slot_nnz_array: List[int], source: str, eval_source: str,
                mlp_unit_nums: List[int]):
    # compile and train model
    if not os.path.exists(model_config_path):
        solver = hugectr.CreateSolver(
            max_eval_batches=512,
            batchsize_eval=256,
            batchsize=256,
            lr=0.001,
            vvgpu=[[0]],
            repeat_dataset=True,
            i64_input_key=False,
        )
        reader = hugectr.DataReaderParams(
            data_reader_type=hugectr.DataReaderType_t.Raw,
            source=[source],
            eval_source=eval_source,
            slot_size_array=row_nums,
            check_type=hugectr.Check_t.Non,
            num_workers=1,
            num_samples=256,
            eval_num_samples=256,
            float_label_dense=False,
        )
        optimizer = hugectr.CreateOptimizer(
            optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
        )
        model = hugectr.Model(solver, reader, optimizer)
        model.add(
            hugectr.Input(
                label_dim=1,
                label_name="label",
                dense_dim=0,
                dense_name="dense",
                data_reader_sparse_param_array=[
                    hugectr.DataReaderSparseParam(
                        "data",
                        slot_nnz_array,
                        False,
                        len(row_nums),
                    )
                ],
            )
        )
        model.add(
            hugectr.SparseEmbedding(
                embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
                slot_size_array=row_nums,
                embedding_vec_size=embed_dim,
                combiner="sum",
                sparse_embedding_name="sparse_embedding",
                bottom_name="data",
                optimizer=optimizer,
            )
        )
        model.add(
            hugectr.DenseLayer(
                layer_type=hugectr.Layer_t.Reshape,
                bottom_names=["sparse_embedding"],
                top_names=["reshape"],
                leading_dim=embed_dim * len(row_nums),
            )
        )
        model.add(
            hugectr.DenseLayer(
                layer_type=hugectr.Layer_t.Concat,
                bottom_names=["reshape"],
                top_names=["concat"],
            )
        )
        unit_nums = list(mlp_unit_nums) + [1]
        for i, units in enumerate(unit_nums):
            model.add(
                hugectr.DenseLayer(
                    layer_type=hugectr.Layer_t.InnerProduct,
                    bottom_names=["concat" if i == 0 else f"fc{i-1}"],
                    top_names=[f"fc{i}"],
                    num_output=units,
                )
            )
        model.add(
            hugectr.DenseLayer(
                layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
                bottom_names=[f"fc{len(unit_nums)-1}", "label"],
                top_names=["loss"],
            )
        )

        model.compile()
        model.summary()
        model.fit(
            max_iter=1024,
            display=200,
            eval_interval=1000,
            snapshot=1000,
            snapshot_prefix=f"{model_name}/{model_name}",
        )
        model.graph_to_json(graph_config_file=model_config_path)


def inference(model_name: str, model_config_path: str, gpu_cache: bool,
              input_batches: List[Tuple[List[int], List[int]]], output: str):
    dense_model_file = f"{model_name}/{model_name}_dense_1000.model"
    sparse_model_files = [f"{model_name}/{model_name}0_sparse_1000.model"]

    inference_params = InferenceParams(
        model_name=model_name,
        max_batchsize=512,
        hit_rate_threshold=1,
        dense_model_file=dense_model_file,
        sparse_model_files=sparse_model_files,
        device_id=0,
        use_gpu_embedding_cache=gpu_cache,
        cache_size_percentage=1,
        i64_input_key=False,
        use_mixed_precision=False,
    )
    inference_session = CreateInferenceSession(model_config_path, inference_params)

    cu_prof = CudaProfiler()
    cu_prof.start()
    accum_time = 0
    for embedding_columns, row_ptrs in input_batches:
        t1 = time.time()
        inference_session.predict([], embedding_columns, row_ptrs)
        t2 = time.time()
        accum_time += t2 - t1
    cu_prof.stop()

    if output:
        with open(output, "w") as f:
            f.write(str(accum_time * 1000))


def main():
    os.environ["HUGECTR_DISABLE_OVERFLOW_CHECK"] = "1"
    args = parse_args()
    os.makedirs(args.model_name, exist_ok=True)
    model_config_path = f"{args.model_name}/config.json"
    source = f"{args.model_name}/data_raw/file_list.txt"
    eval_source = f"{args.model_name}/data_raw/file_list_test.txt"

    table_shapes = parse_table_shapes(args.table_config_path)
    row_nums = [row_num for row_num, embed_dim in table_shapes]
    embed_dim = table_shapes[0][1]
    for row_num, cur_embed_dim in table_shapes:
        assert embed_dim == cur_embed_dim, f"{embed_dim} != {cur_embed_dim}"

    slot_nnz_array = parse_slot_nnz_array(args.data_config_path)
    generate_train_data(row_nums, source, eval_source)
    build_model(args.model_name, model_config_path, row_nums, embed_dim, slot_nnz_array,
                source, eval_source, args.mlp_unit_nums)

    input_batches = parse_sparse_data(args.data_dir, row_nums)
    inference(args.model_name, model_config_path, args.gpu_cache, input_batches, args.output)


if __name__ == "__main__":
    main()
