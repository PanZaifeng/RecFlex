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
import multiprocessing as mp
from typing import List

from RecFlex import TaskManager
from RecFlex.cuda_helper import CudaInfoQuerier
from RecFlex.parser import parse_knl_total_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--model_base_dir", type=str, required=True)
    parser.add_argument("-m", "--model_names", nargs="+", type=str, required=True)
    parser.add_argument("-l", "--librecom_path", type=str)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--remeasure", action="store_true", default=False)
    args = parser.parse_args()
    return args

def nsys_profile(data_dir: str, data_config_path: str, table_config_path: str, report_prefix: str,
                 librecom_path: str, gpu_id: int = None):
    if gpu_id is None:
        raise Exception("No vaild GPU provided!")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(cur_dir, "benchmark_tensorflow.py")

    profile_prefix = f"nsys profile -c cudaProfilerApi -t cuda -f true -o {report_prefix}"
    launch_script = (
        f"{sys.executable} {script_path} "
        f"--data_config_path {data_config_path} "
        f"--data_dir {data_dir} "
        f"--table_config_path {table_config_path} "
    )
    if librecom_path:
        launch_script += f"--lib_path {librecom_path} "
    # nsys will return non-zero
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {profile_prefix} {launch_script}")

def launch_tasks(args: argparse.Namespace, gpu_ids: List[int]):
    task_manager = TaskManager([gpu_ids[0]])
    for model_name in args.model_names:
        model_dir = os.path.join(args.model_base_dir, model_name)
        data_dir = os.path.join(model_dir, "data_test")
        data_config_path = os.path.join(model_dir, "data_config.txt")
        table_config_path = os.path.join(model_dir, "table_config.txt")
        report_prefix = os.path.join(args.output_dir, model_name)
        report_name = f"{report_prefix}.nsys-rep"
        if not os.path.exists(report_name) or args.remeasure:
            task_manager.add_task(nsys_profile, (data_dir, data_config_path, table_config_path,
                                                 report_prefix, args.librecom_path),
                                  ident_key="gpu_id")
    task_manager.join()

    with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
        for model_name in args.model_names:
            report_prefix = os.path.join(args.output_dir, model_name)
            report_name = f"{report_prefix}.nsys-rep"
            if args.librecom_path:
                time = parse_knl_total_time("FusedKnl", report_name)
            else:
                time = 0
                time += parse_knl_total_time("Segment", report_name)
                time += parse_knl_total_time("Gather", report_name)
            f.write(f"{model_name},{time/1e6}\n")

def main():
    args = parse_args()
    querier = CudaInfoQuerier()
    gpu_ids = querier.gpu_ids
    os.makedirs(args.output_dir, exist_ok=True)
    launch_tasks(args, gpu_ids)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
