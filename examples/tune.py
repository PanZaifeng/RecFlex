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

from RecFlex.tuning import Tuner, TuneConfig, BasicConfig
from RecFlex.utils import get_data_fnames, compile_build
from RecFlex.cuda_helper import CudaInfoQuerier

from schedules.reduce_by_key import ReduceByKeySchedule, ReduceByKeyParams
from schedules.multi_block_per_sample import MultiBlockPerSampleSchedule, MultiBlockPerSampleParams
from schedules.one_block_multi_sample import OneBlockMultiSampleSchedule, OneBlockMultiSampleParams
from schedules.warp_per_sample import WarpPerSampleSchedule, WarpPerSampleParams
from schedules.one_hot import OneHotSchedule, OneHotParams
from schedules.one_hot_multi_block import OneHotMultiBlockSchedule, OneHotMultiBlockParams

import argparse
import os
import multiprocessing as mp
import numpy as np


def get_csrc_dir() -> str:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(cur_dir, "csrc")
    return csrc_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-n", "--num_processes", type=int, default=64)
    parser.add_argument("-b", "--block_threads", type=int, default=256)
    parser.add_argument("--mutation_feat_id", type=int)
    parser.add_argument("--naive", action="store_true", default=False)
    parser.add_argument("--fixed_thread_binding", action="store_true", default=False)
    parser.add_argument("--thread_binding_choice", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--recompile", action="store_true", default=False)
    parser.add_argument("--remeasure", action="store_true", default=False)
    parser.add_argument("--noinline", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def create_tuner(args: argparse.Namespace) -> Tuner:
    model_dir = args.model_dir
    data_dir = os.path.join(model_dir, "data")
    data_fnames = get_data_fnames(data_dir)
    csrc_dir = get_csrc_dir()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(model_dir, "output")

    libschedule_build_dir = os.path.join(csrc_dir, "build")
    libschedule_path = os.path.join(libschedule_build_dir, "libschedules.so")
    if not os.path.exists(libschedule_path):
        compile_build(libschedule_build_dir)

    basic_config = BasicConfig(batch_sizes_fname=os.path.join(data_dir, "batch_sizes.txt"),
                               data_fnames=data_fnames,
                               schedule_config_fname=os.path.join(model_dir, "schedule_config.txt"),
                               table_config_fname=os.path.join(model_dir, "table_config.txt"),
                               codegen_dir=output_dir,
                               block_threads=args.block_threads,
                               includes=[os.path.join(csrc_dir, "include")],
                               links=[libschedule_path],
                               host_links=[libschedule_path])
    tuner = Tuner(basic_config=basic_config, schedule_mapping=globals())
    return tuner


def create_tune_config(args: argparse.Namespace) -> TuneConfig:
    querier = CudaInfoQuerier()
    cc_major, cc_minor = querier.get_compute_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc_major}.{cc_minor}"

    gpu_ids = querier.gpu_ids
    sm_count, max_threads_per_sm = querier.get_sm_thread_count()
    occupancies = list(range(1, max_threads_per_sm // args.block_threads + 1))
    tune_config = TuneConfig(occupancies=occupancies, gpu_ids=gpu_ids, sm_count=sm_count,
                             num_processes=args.num_processes, recompile=args.recompile,
                             remeasure=args.remeasure, inline=not args.noinline, debug=args.debug)
    return tune_config


def main():
    args = parse_args()
    tuner = create_tuner(args)
    tune_config = create_tune_config(args)

    if not args.naive:
        tune_results = tuner.tune(tune_config)
        print(tune_results)
        if not args.local_only:
            occupancy, sched_params = tuner.solve(tune_results, tune_config)
            print(occupancy, sched_params)

            if args.fixed_thread_binding:
                fixed_task_blocks, t = tuner.tune_fixed_thread_binding(opt_sched_params=sched_params,
                                                                       choice=args.thread_binding_choice,
                                                                       tune_config=tune_config)
                print(fixed_task_blocks)
                print(f"Fixed thread binding result: {t/1e6} ms")

            if args.mutation_feat_id is not None and args.mutation_feat_id >= 0:
                mut_times, opt_idx = tuner.mutation_validation(opt_sched_params=sched_params, opt_occupancy=occupancy,
                                                               mut_feat_id=args.mutation_feat_id, tune_config=tune_config)
                print(mut_times, np.argmin(mut_times), opt_idx)
    else:
        tune_results, t = tuner.tune_naive(tune_config)
        print(tune_results)
        print(f"Naive-tuning result: {t/1e6} ms")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
