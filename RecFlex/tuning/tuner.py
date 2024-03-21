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
import numpy as np

from typing import Dict, List, Tuple, NamedTuple
from RecFlex.code_emitter import CodeEmitter, ScheduleBase, MetaData
from RecFlex.utils import find_file, compile_build, os_check
from RecFlex.task_manager import TaskManager
from RecFlex.parser import parse_raw_data_batches, parse_table_shapes, parse_schedule_types, parse_knl_total_time

from .measure import measure_single_feat, nsys_profile, measure_task_blocks, measure_naive_candidate


class BasicConfig(NamedTuple):
    batch_sizes_fname: str
    data_fnames: List[str]
    schedule_config_fname: str
    table_config_fname: str
    codegen_dir: str
    subdirs: str = []
    includes: str = []
    links: str = []
    host_links: str = []
    block_threads: int = 256
    sample_batch_nums: int = 128


class TuneConfig(NamedTuple):
    occupancies: List[int]
    gpu_ids: List[int]
    sm_count: int
    num_processes: int = 8
    inline: bool = True
    recompile: bool = False
    remeasure: bool = False
    debug: bool = False


class Tuner:
    def __init__(self, basic_config: BasicConfig, schedule_mapping: Dict[str, ScheduleBase]):
        self.basic = basic_config
        self.schedule_mapping = schedule_mapping
        self.get_raw_data_batches()
        self.get_metas()
        self.get_feat_schedules()
        self.num_feats: int = len(self.feat_schedules)
        self.gen_feat_sched_param_candidates()

    def get_raw_data_batches(self):
        self.raw_data_batches = parse_raw_data_batches(self.basic.batch_sizes_fname,
                                                       self.basic.data_fnames,
                                                       self.basic.sample_batch_nums)
        return self.raw_data_batches

    def get_metas(self):
        self.table_shapes = parse_table_shapes(self.basic.table_config_fname)
        self.metas: List[MetaData] = [MetaData(embed_dim=embed_dim) for nrows, embed_dim in self.table_shapes]
        return self.metas

    def get_feat_schedules(self):
        feat_schedule_types = parse_schedule_types(self.basic.schedule_config_fname, self.schedule_mapping)
        self.feat_schedules: List[List[ScheduleBase]] = [[schedule_type(meta, self.basic.block_threads) for schedule_type in schedule_types]
                                                         for schedule_types, meta in zip(feat_schedule_types, self.metas)]
        return self.feat_schedules

    def gen_feat_sched_param_candidates(self):
        self.feat_sched_param_candidates: List[List[Tuple[ScheduleBase, NamedTuple]]] = [[] for _ in range(self.num_feats)]
        for feat_id in range(self.num_feats):
            for schedule in self.feat_schedules[feat_id]:
                for params in schedule.GenParamCandidates():
                    self.feat_sched_param_candidates[feat_id].append([schedule, params])

    def get_padded_sched_num(self, orig_num: int, max_concurrent_blocks: int) -> int:
        return (orig_num + max_concurrent_blocks - 1) // orig_num * orig_num

    def pad_sched_params(self, sched_params: List[Tuple[ScheduleBase, NamedTuple]], padded_num: int) -> List[Tuple[ScheduleBase, NamedTuple]]:
        padded_schedules = sched_params.copy()
        add_num = padded_num - len(sched_params)
        for i in range(add_num):
            padded_schedules.append(sched_params[i % len(sched_params)])
        return padded_schedules

    def get_output_build_dir(self, feat_id: int, occupancy: int) -> Tuple[str, str]:
        if feat_id >= 0:
            output_dir = f"{self.basic.codegen_dir}/f{feat_id}/o{occupancy}"
        else:
            output_dir = f"{self.basic.codegen_dir}/all/o{occupancy}"
        build_dir = f"{output_dir}/build"
        return output_dir, build_dir

    def emit_codes(self, output_dir: str, feat_schedules: List[Tuple[ScheduleBase, NamedTuple]],
                   occupancy: int, inline: bool = True, timing: bool = False, tune: bool = False,
                   fixed_task_block_nums: List[int] = None, debug: bool = False) -> None:
        emitter = CodeEmitter(feat_schedules)
        emitter.emit_host(f"{output_dir}/preprocess.cu", fixed_task_block_nums=fixed_task_block_nums)
        emitter.emit(f"{output_dir}/process.cu", occupancy=occupancy, timing=timing, inline=inline, tune=tune,
                     fixed_task_block_nums=fixed_task_block_nums)
        emitter.emit_cmake_list(f"{output_dir}/CMakeLists.txt", code_fname="process.cu", host_fname="preprocess.cu",
                                subdirs=self.basic.subdirs, includes=self.basic.includes, links=self.basic.links,
                                host_links=self.basic.host_links, debug=debug)

    def tune(self, tune_config: TuneConfig) -> Dict[int, Dict[int, List[int]]]:
        compile_task_manager = TaskManager(list(range(tune_config.num_processes)))
        for feat_id in range(self.num_feats):
            for occupancy in tune_config.occupancies:
                output_dir, build_dir = self.get_output_build_dir(feat_id, occupancy)
                if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
                    os.makedirs(build_dir, exist_ok=True)
                    padded_sched_num = self.get_padded_sched_num(len(self.feat_sched_param_candidates[feat_id]),
                                                                 tune_config.sm_count * occupancy)
                    padded_sched_param_candidates = self.pad_sched_params(self.feat_sched_param_candidates[feat_id],
                                                                          padded_sched_num)
                    self.emit_codes(output_dir=output_dir, feat_schedules=padded_sched_param_candidates, occupancy=occupancy,
                                    inline=tune_config.inline, timing=True, tune=True, debug=tune_config.debug)
                    compile_task_manager.add_task(compile_build, (build_dir,))
        compile_task_manager.join()

        measure_task_manager = TaskManager(tune_config.gpu_ids)
        for feat_id in range(self.num_feats):
            for occupancy in tune_config.occupancies:
                output_dir, build_dir = self.get_output_build_dir(feat_id, occupancy)
                result_log = f"{output_dir}/result.txt"
                if not os.path.exists(result_log) or tune_config.remeasure:
                    padded_sched_num = self.get_padded_sched_num(len(self.feat_sched_param_candidates[feat_id]),
                                                                 tune_config.sm_count * occupancy)
                    measure_task_manager.add_task(measure_single_feat, ((feat_id, occupancy), self.raw_data_batches[feat_id],
                                                                        self.table_shapes[feat_id], len(self.feat_sched_param_candidates[feat_id]),
                                                                        padded_sched_num, f"{build_dir}/librecom.so",
                                                                        find_file(build_dir, "recom_host*.so"), result_log),
                                                  ident_key="gpu_id")
        measure_task_manager.join()

        results: Dict[int, Dict[int, List[int]]] = {feat_id: {} for feat_id in range(self.num_feats)}
        for feat_id in range(self.num_feats):
            for occupancy in tune_config.occupancies:
                output_dir, build_dir = self.get_output_build_dir(feat_id, occupancy)
                result_log = f"{output_dir}/result.txt"
                with open(result_log) as f:
                    result = list(map(float, f.read().strip().split(' ')))
                results[feat_id][occupancy] = result

        return results

    def solve(self, measured_times: Dict[int, Dict[int, List[int]]], tune_config: TuneConfig) -> Tuple[int, List[Tuple[ScheduleBase, NamedTuple]]]:
        occu_opt_sched_params: List[List[Tuple[ScheduleBase, NamedTuple]]] = [[] for _ in tune_config.occupancies]
        for feat_id in range(self.num_feats):
            for i, occupancy in enumerate(tune_config.occupancies):
                opt_idx = np.argmin(measured_times[feat_id][occupancy])
                opt_sched_param = self.feat_sched_param_candidates[feat_id][opt_idx]
                occu_opt_sched_params[i].append(opt_sched_param)

        opt_occupancy, opt_sched_params = self.tune_occupancy(occu_opt_sched_params, tune_config)
        output_dir, build_dir = self.get_output_build_dir(-1, opt_occupancy)
        os_check(f"cp -r {output_dir} {self.basic.codegen_dir}/optimal")

        return opt_occupancy, opt_sched_params

    def tune_occupancy(self, occu_opt_sched_params: List[List[Tuple[ScheduleBase, NamedTuple]]],
                       tune_config: TuneConfig) -> Tuple[int, List[Tuple[ScheduleBase, NamedTuple]]]:
        compile_task_threads = min([tune_config.num_processes, len(tune_config.occupancies)])
        compile_task_manager = TaskManager(list(range(compile_task_threads)))
        for occupancy, sched_params in zip(tune_config.occupancies, occu_opt_sched_params):
            output_dir, build_dir = self.get_output_build_dir(-1, occupancy)
            if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
                os.makedirs(build_dir, exist_ok=True)
                self.emit_codes(output_dir=output_dir, feat_schedules=sched_params, occupancy=occupancy,
                                inline=tune_config.inline, timing=False, tune=False, debug=tune_config.debug)
                compile_task_manager.add_task(compile_build, (build_dir,))
        compile_task_manager.join()

        measure_task_manager = TaskManager(tune_config.gpu_ids[0:1])
        for occupancy in tune_config.occupancies:
            output_dir, build_dir = self.get_output_build_dir(-1, occupancy)
            report_prefix = f"{output_dir}/report"
            report_name = f"{report_prefix}.nsys-rep"
            if not os.path.exists(report_name) or tune_config.remeasure:
                os_check(f"rm -f {report_name}")
                measure_task_manager.add_task(nsys_profile, (self.basic.table_config_fname, build_dir,
                                                             self.basic.data_fnames, self.basic.sample_batch_nums,
                                                             report_prefix),
                                              ident_key="gpu_id")
        measure_task_manager.join()

        times = []
        for occupancy in tune_config.occupancies:
            output_dir, build_dir = self.get_output_build_dir(-1, occupancy)
            t = parse_knl_total_time("FusedKnl", f"{output_dir}/report.nsys-rep")
            times.append(t)
            print(f"occupancy: {occupancy}, t(ns): {t}")

        opt_idx = np.argmin(times)
        opt_occupancy = tune_config.occupancies[opt_idx]
        opt_sched_params = occu_opt_sched_params[opt_idx]

        return opt_occupancy, opt_sched_params

    def mutation_validation(self, opt_sched_params: List[Tuple[ScheduleBase, NamedTuple]], opt_occupancy: int,
                            mut_feat_id: int, tune_config: TuneConfig) -> Tuple[List[int], int]:
        compile_task_threads = min([tune_config.num_processes, len(tune_config.occupancies)])
        compile_task_manager = TaskManager(list(range(compile_task_threads)))
        mut_sched_params = opt_sched_params.copy()
        mutuation_dir = f"{self.basic.codegen_dir}/mutation_f{mut_feat_id}"
        for candidate_id, mut_sched_param in enumerate(self.feat_sched_param_candidates[mut_feat_id]):
            if mut_sched_param == opt_sched_params[mut_feat_id]:
                opt_idx = candidate_id
            mut_sched_params[mut_feat_id] = mut_sched_param
            output_dir = f"{mutuation_dir}/m{candidate_id}"
            build_dir = f"{output_dir}/build"
            if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
                os.makedirs(build_dir, exist_ok=True)
                self.emit_codes(output_dir=output_dir, feat_schedules=mut_sched_params,
                                occupancy=opt_occupancy, inline=tune_config.inline,
                                timing=False, tune=False, debug=tune_config.debug)
                compile_task_manager.add_task(compile_build, (build_dir,))
        compile_task_manager.join()

        measure_task_manager = TaskManager(tune_config.gpu_ids[0:1])
        for candidate_id, mut_sched_param in enumerate(self.feat_sched_param_candidates[mut_feat_id]):
            output_dir = f"{mutuation_dir}/m{candidate_id}"
            build_dir = f"{output_dir}/build"
            report_prefix = f"{output_dir}/report"
            report_name = f"{report_prefix}.nsys-rep"
            if not os.path.exists(report_name) or tune_config.remeasure:
                os_check(f"rm -f {report_prefix}.*")
                measure_task_manager.add_task(nsys_profile, (self.basic.table_config_fname, build_dir,
                                                             self.basic.data_fnames, self.basic.sample_batch_nums,
                                                             report_prefix),
                                              ident_key="gpu_id")
        measure_task_manager.join()

        times = []
        for candidate_id, mut_sched_param in enumerate(self.feat_sched_param_candidates[mut_feat_id]):
            output_dir = f"{mutuation_dir}/m{candidate_id}"
            t = parse_knl_total_time("FusedKnl", f"{output_dir}/report.nsys-rep")
            times.append(t)

        return times, opt_idx

    def tune_naive(self, tune_config: TuneConfig) -> Tuple[List[Tuple[ScheduleBase, NamedTuple]], int]:
        def get_output_build_dir(feat_id: int, sched_param_id: int):
            output_dir = f"{self.basic.codegen_dir}/naive/f{feat_id}/s{sched_param_id}"
            build_dir = f"{output_dir}/build"
            return output_dir, build_dir

        compile_task_manager = TaskManager(list(range(tune_config.num_processes)))
        for feat_id in range(self.num_feats):
            for sched_param_id, sched_param in enumerate(self.feat_sched_param_candidates[feat_id]):
                output_dir, build_dir = get_output_build_dir(feat_id, sched_param_id)
                if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
                    os.makedirs(build_dir, exist_ok=True)
                    self.emit_codes(output_dir=output_dir, feat_schedules=[sched_param], occupancy=None,
                                    inline=tune_config.inline, timing=True, tune=False, debug=tune_config.debug)
                    compile_task_manager.add_task(compile_build, (build_dir,))
        compile_task_manager.join()

        measure_task_manager = TaskManager(tune_config.gpu_ids)
        for feat_id in range(self.num_feats):
            for sched_param_id, sched_param in enumerate(self.feat_sched_param_candidates[feat_id]):
                output_dir, build_dir = get_output_build_dir(feat_id, sched_param_id)
                result_log = f"{output_dir}/result.txt"
                if not os.path.exists(result_log) or tune_config.remeasure:
                    measure_task_manager.add_task(measure_naive_candidate, ((feat_id, sched_param_id), self.raw_data_batches[feat_id],
                                                                            self.table_shapes[feat_id], f"{build_dir}/librecom.so",
                                                                            find_file(build_dir, "recom_host*.so"), result_log),
                                                  ident_key="gpu_id")
        measure_task_manager.join()

        opt_sched_params: List[Tuple[ScheduleBase, NamedTuple]] = []
        for feat_id in range(self.num_feats):
            times = []
            for sched_param_id, sched_param in enumerate(self.feat_sched_param_candidates[feat_id]):
                output_dir, build_dir = get_output_build_dir(feat_id, sched_param_id)
                result_log = f"{output_dir}/result.txt"
                with open(result_log) as f:
                    t = float(f.read().strip())
                times.append(t)
            opt_idx = np.argmin(times)
            opt_sched_params.append(self.feat_sched_param_candidates[feat_id][opt_idx])

        output_dir = f"{self.basic.codegen_dir}/naive/optimal"
        build_dir = f"{output_dir}/build"
        if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
            os.makedirs(build_dir, exist_ok=True)
            self.emit_codes(output_dir=output_dir, feat_schedules=opt_sched_params, occupancy=None,
                            inline=tune_config.inline, timing=False, tune=False, debug=tune_config.debug)
            compile_task_manager.add_task(compile_build, (build_dir,))
            compile_task_manager.join()

        report_prefix = f"{output_dir}/report"
        report_name = f"{report_prefix}.nsys-rep"
        if not os.path.exists(report_name) or tune_config.remeasure:
            os_check(f"rm -f {report_prefix}.*")
            measure_task_manager.add_task(nsys_profile, (self.basic.table_config_fname, build_dir,
                                                         self.basic.data_fnames, self.basic.sample_batch_nums,
                                                         report_prefix),
                                          ident_key="gpu_id")
            measure_task_manager.join()

        opt_time = parse_knl_total_time("FusedKnl", f"{output_dir}/report.nsys-rep")
        return opt_sched_params, opt_time

    def tune_fixed_thread_binding(self, opt_sched_params: List[Tuple[ScheduleBase, NamedTuple]], choice: str,
                                  tune_config: TuneConfig) -> Tuple[List[int], int]:
        assert choice in ["max", "mean"]
        output_dir = f"{self.basic.codegen_dir}/fixed_thread_binding_{choice}"
        fixed_blocks_path = f"{output_dir}/task_blocks.txt"
        if not os.path.exists(fixed_blocks_path) or tune_config.remeasure:
            os.makedirs(output_dir, exist_ok=True)
            opt_dir = f"{self.basic.codegen_dir}/optimal"
            opt_build_dir = f"{opt_dir}/build"
            measure_task_manager = TaskManager([0])
            measure_task_manager.add_task(measure_task_blocks, (self.raw_data_batches, f"{opt_build_dir}/librecom.so",
                                                                find_file(opt_build_dir, "recom_host*.so"), fixed_blocks_path))
            measure_task_manager.join()

        task_block_batches: List[List[int]] = []
        with open(fixed_blocks_path) as f:
            for line in f.readlines():
                task_block_batches.append(list(map(int, line.strip().split(','))))

        if choice == "max":
            fixed_task_blocks: List[int] = np.max(task_block_batches, axis=0).tolist()
        else:
            fixed_task_blocks: List[int] = np.mean(task_block_batches, axis=0, dtype=np.int32).tolist()

        build_dir = f"{output_dir}/build"
        if not os.path.exists(f"{build_dir}/librecom.so") or tune_config.recompile:
            os.makedirs(build_dir, exist_ok=True)
            compile_task_manager = TaskManager([0])
            self.emit_codes(output_dir=output_dir, feat_schedules=opt_sched_params, fixed_task_block_nums=fixed_task_blocks,
                            occupancy=None, inline=tune_config.inline, timing=False, tune=False, debug=tune_config.debug)
            compile_task_manager.add_task(compile_build, (build_dir,))
            compile_task_manager.join()

        report_prefix = f"{output_dir}/report"
        report_name = f"{report_prefix}.nsys-rep"
        if not os.path.exists(report_name) or tune_config.remeasure:
            os_check(f"rm -f {report_prefix}.*")
            measure_task_manager = TaskManager(tune_config.gpu_ids[0:1])
            measure_task_manager.add_task(nsys_profile, (self.basic.table_config_fname, build_dir,
                                                         self.basic.data_fnames, self.basic.sample_batch_nums,
                                                         report_prefix),
                                          ident_key="gpu_id")
            measure_task_manager.join()

        opt_time = parse_knl_total_time("FusedKnl", f"{output_dir}/report.nsys-rep")
        return fixed_task_blocks, opt_time
