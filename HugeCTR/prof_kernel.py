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
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-e", "--e2e", action="store_true", default=False)
    parser.add_argument("-g", "--gpu_cache", action="store_true", default=False)
    parser.add_argument("--remeasure", action="store_true", default=False)
    args = parser.parse_args()
    return args


def nsys_profile(model_name: str, data_dir: str, data_config_path: str, table_config_path: str, rep_prefix_or_fname: str,
                 e2e: bool, gpu_cache: bool, gpu_id: int = None):
    if gpu_id is None:
        raise Exception("No vaild GPU provided!")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(cur_dir, "benchmark_hugectr.py")

    launch_script = (
        f"{sys.executable} {script_path} "
        f"--model_name {model_name} "
        f"--data_dir {data_dir} "
        f"--data_config_path {data_config_path} "
        f"--table_config_path {table_config_path} "
    )

    if gpu_cache:
        launch_script += "--gpu_cache "

    if e2e:
        profile_prefix = ""
        launch_script += f"--output {rep_prefix_or_fname}"
    else:
        profile_prefix = f"nsys profile -c cudaProfilerApi -t cuda -f true -o {rep_prefix_or_fname}"

    # nsys will return non-zero
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {profile_prefix} {launch_script}")


def launch_tasks(args: argparse.Namespace, gpu_ids: List[int]):
    task_manager = TaskManager(gpu_ids)
    iname = 'g' if args.gpu_cache else 'c'
    for model_name in args.model_names:
        model_dir = os.path.join(args.model_base_dir, model_name)
        data_dir = os.path.join(model_dir, "data")
        data_config_path = os.path.join(model_dir, "data_config.txt")
        table_config_path = os.path.join(model_dir, "table_config.txt")
        report_prefix = os.path.join(args.output_dir, f"{model_name}_{iname}")
        if args.e2e:
            result_path = f"{report_prefix}_e2e.txt"
            if not os.path.exists(result_path) or args.remeasure:
                task_manager.add_task(nsys_profile, (model_name, data_dir, data_config_path,
                                                     table_config_path, result_path,
                                                     args.e2e, args.gpu_cache),
                                      ident_key="gpu_id")
        else:
            report_name = f"{report_prefix}.nsys-rep"
            if not os.path.exists(report_name) or args.remeasure:
                task_manager.add_task(nsys_profile, (model_name, data_dir, data_config_path,
                                                     table_config_path, report_prefix,
                                                     args.e2e, args.gpu_cache),
                                      ident_key="gpu_id")
    task_manager.join()

    if args.e2e:
        with open(os.path.join(args.output_dir, f"result_{iname}_e2e.txt"), "w") as f:
            for model_name in args.model_names:
                e2e_path = os.path.join(args.output_dir, f"{model_name}_{iname}_e2e.txt")
                with open(e2e_path) as res:
                    t = res.read().strip()
                    f.write(f"{model_name},{t}\n")
    else:
        with open(os.path.join(args.output_dir, f"result_{iname}.txt"), "w") as f:
            for model_name in args.model_names:
                report_prefix = os.path.join(args.output_dir, f"{model_name}_{iname}")
                report_name = f"{report_prefix}.nsys-rep"
                t = parse_knl_total_time("embedding", report_name)
                f.write(f"{model_name},{t/1e6}\n")


def main():
    args = parse_args()
    querier = CudaInfoQuerier()
    gpu_ids = querier.gpu_ids
    os.makedirs(args.output_dir, exist_ok=True)
    launch_tasks(args, gpu_ids)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
