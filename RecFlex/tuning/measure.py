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

import os
import sys
from typing import List, Tuple

from RecFlex.rec_modules import FeatProcessEmbed
from RecFlex.utils import import_from_path, os_check


def measure_single_feat(ident: Tuple[int, int], raw_data_batches_single_feat: List[List[str]],
                        table_shape: Tuple[int, int], num_schedules: int, num_padded_schedules: int,
                        librecom_path: str, recom_host_path: str, output_path: str, gpu_id: int = None) -> List[float]:
    # We have to use a new process to call this function to avoid
    # namespace conflicts of Torch libraries
    if gpu_id is None:
        raise Exception("No vaild GPU provided!")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    feat_id, occupancy = ident
    print(f"Feat {feat_id}, Occ {occupancy}, GPU {gpu_id}")

    result = None
    try:
        import torch
        assert torch.cuda.is_available()

        recom_host = import_from_path("recom_host", recom_host_path)
        torch.ops.load_library(librecom_path)

        cuda_dev = torch.cuda.current_device()
        embed_table = torch.ones(size=table_shape, dtype=torch.float32, device=cuda_dev)
        embed_tables = [embed_table for _ in range(num_padded_schedules)]

        model = FeatProcessEmbed(recom_host=recom_host, embed_tables=embed_tables, device=cuda_dev)

        accum_times = torch.zeros([num_schedules], dtype=torch.float64)
        for raw_data_single_feat in raw_data_batches_single_feat:
            raw_data = [raw_data_single_feat for _ in range(num_padded_schedules)]
            output, task_times = model(raw_data, timing=True, timing_schedules=num_schedules)
            accum_times += task_times

        result = ' '.join([str(t.item()) for t in accum_times])
        temp = output_path + ".tmp"
        with open(temp, "w") as f:
            f.write(result)
        os_check(f"mv {temp} {output_path}")
    except Exception as e:
        print(f"[{ident}] Execption {e}")


def measure_naive_candidate(ident: Tuple[int, int], raw_data_batches_single_feat: List[List[str]],
                            table_shape: Tuple[int, int], librecom_path: str, recom_host_path: str,
                            output_path: str, gpu_id: int = None) -> List[float]:
    # We have to use a new process to call this function to avoid
    # namespace conflicts of Torch libraries
    if gpu_id is None:
        raise Exception("No vaild GPU provided!")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    feat_id, sched_param_id = ident
    print(f"Feat {feat_id}, Schedule {sched_param_id}, GPU {gpu_id}")

    try:
        import torch
        assert torch.cuda.is_available()

        recom_host = import_from_path("recom_host", recom_host_path)
        torch.ops.load_library(librecom_path)

        cuda_dev = torch.cuda.current_device()
        embed_table = torch.ones(size=table_shape, dtype=torch.float32, device=cuda_dev)
        embed_tables = [embed_table]

        model = FeatProcessEmbed(recom_host=recom_host, embed_tables=embed_tables, device=cuda_dev)

        accum_time = 0
        for raw_data_single_feat in raw_data_batches_single_feat:
            raw_data = [raw_data_single_feat]
            output, t = model(raw_data, timing_kernel=True)
            accum_time += t

        with open(output_path, "w") as f:
            f.write(str(accum_time))
    except Exception as e:
        print(f"[{ident}] Execption {e}")


def nsys_profile(table_config_fname: str, build_dir: str, data_fnames: str, sample_batch_nums: int,
                 report_prefix: str, spec_feat_id: int = None, gpu_id: int = None):
    if gpu_id is None:
        raise Exception("No vaild GPU provided!")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(cur_dir, "run_script.py")

    data_dir = os.path.dirname(data_fnames[0])
    if not spec_feat_id:
        for feat_id, data_fname in enumerate(data_fnames):
            assert data_fname == f"{data_dir}/f{feat_id}.txt", f"Currently not support flexible data file names!"

    profile_prefix = f"nsys profile -c cudaProfilerApi -t cuda -f true -o {report_prefix}"
    launch_script = (
        f"{sys.executable} {script_path} "
        f"--table_config_fname {table_config_fname} "
        f"--build_dir {build_dir} "
        f"--data_dir {data_dir} "
        f"--sample_batch_nums {sample_batch_nums} "
    )
    if spec_feat_id:
        launch_script += f"--spec_feat_id {spec_feat_id} "
    # nsys will return non-zero
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {profile_prefix} {launch_script}")


def measure_task_blocks(raw_data_batches: List[List[str]], librecom_path: str, recom_host_path: str,
                        output_path: str):
    import torch
    torch.ops.load_library(librecom_path)
    recom_host = import_from_path("recom_host", recom_host_path)

    num_feats = len(raw_data_batches)
    num_batches = len(raw_data_batches[0])
    raw_data_batches_trans: List[List[str]] = [list(row) for row in zip(*raw_data_batches)]

    max_concurrent_blocks = torch.ops.recom.get_max_concurrent_blocks()
    task_block_batches = []
    for raw_data in raw_data_batches_trans:
        data_buffer_offsets, output_offsets, extra_buffer_offsets, task_mapping, task_blocks, data, scala_data = recom_host.preprocess(raw_data, max_concurrent_blocks)
        # TODO: there are some bugs to fix in task_blocks/task_tiles codegen; workaround here
        task_blocks = [0] * num_feats
        for feat_id, task_bid in task_mapping:
            task_blocks[feat_id] += 1
        task_block_batches.append(task_blocks)

    with open(output_path, "w") as f:
        for task_blocks in task_block_batches:
            f.write(','.join([str(b) for b in task_blocks]) + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feat_id", type=int, required=True)
    parser.add_argument("-g", "--occupancy", type=int, required=True)
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-b", "--build_dir", type=str, required=True)
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    parser.add_argument("--sample_batch_nums", type=int, default=1)
    args = parser.parse_args()

    feat_id = args.feat_id
    occupancy = args.occupancy
    build_dir = args.build_dir
    data_dir = args.data_dir

    from RecFlex.cuda_helper import CudaInfoQuerier
    querier = CudaInfoQuerier()
    sm_count, max_threads_per_sm = querier.get_sm_thread_count()
    gpu_ids = querier.gpu_ids

    from RecFlex.parser import parse_raw_data_batches, parse_table_shapes
    raw_data_batches = parse_raw_data_batches(f"{data_dir}/batch_sizes.txt", [f"{data_dir}/f{feat_id}.txt"],
                                              args.sample_batch_nums)
    table_shapes = parse_table_shapes(args.table_config_path)

    from RecFlex.utils import find_file
    measure_single_feat((feat_id, occupancy), raw_data_batches[0],
                        table_shapes[feat_id], 1, sm_count * occupancy,
                        f"{build_dir}/librecom.so", find_file(build_dir, "recom_host*.so"),
                        "/tmp/result.log", gpu_ids[0])
