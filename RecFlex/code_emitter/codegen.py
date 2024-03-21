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

import json
import os
from typing import List, NamedTuple, Tuple, Dict, Any

from .schedule_base import ScheduleBase, ScheduleVars, PreprocessVars
from .build_basic import BasicBuilder


class CodeEmitter:
    def __init__(self, feat_schedules: List[Tuple[ScheduleBase, NamedTuple]]):
        self.feat_schedules: List[Tuple[ScheduleBase, NamedTuple]] = feat_schedules
        if len(self.feat_schedules) > 0:
            self.block_threads: int = feat_schedules[0][0].block_threads
            for schedule, params in feat_schedules:
                assert self.block_threads == schedule.block_threads

    def serialize_config(self) -> List[Dict[str, Any]]:
        return [
            {
                "schedule": schedule.__class__.__name__,
                "params": params._asdict()
            }
            for schedule, params in self.feat_schedules
        ]

    def dump_key(self, fname: str) -> None:
        with open(fname, "w") as f:
            json.dump(self.serialize(), fp=f)

    def emit_headers(self) -> str:
        headers = {"<cstdlib>", "<cstdio>", "<tuple>",
                   "<torch/extension.h>",
                   "<pybind11/pybind11.h>", "<pybind11/stl.h>",
                   "\"utils.cuh\"", "\"gpu_pointers_array.h\""}
        for schedule, params in self.feat_schedules:
            headers = headers.union(schedule.EmitHeaders())
        return ''.join(sorted([
            f"#include {header}\n"
            for header in headers
        ])) + '\n'

    def emit_host_op_headers(self) -> str:
        headers = {"<cstdlib>", "<vector>", "<thread>", "<string>",
                   "<functional>", "<tuple>", "<iostream>",
                   "<pybind11/pybind11.h>", "<pybind11/stl.h>",
                   "\"utils.cuh\"", "\"task_mapping.cuh\""}
        for schedule, params in self.feat_schedules:
            headers = headers.union(schedule.EmitPrepropHeaders())
        return ''.join(sorted([
            f"#include {header}\n"
            for header in headers
        ]))

    def emit_host_preprocess(self, fixed_task_block_nums: List[int] = None) -> str:
        num_feats = len(self.feat_schedules)
        emitted_code_id_map: Dict[str, int] = {}
        emitted_id_map: Dict[int, int] = {}

        vars_ = PreprocessVars()
        functions = ""
        for feat_idx, (schedule, params) in enumerate(self.feat_schedules):
            host_preprocess = schedule.EmitHostPreprocess(params)
            if host_preprocess in emitted_code_id_map.keys():
                emitted_id_map[feat_idx] = emitted_code_id_map[host_preprocess]
            else:
                emitted_code_id_map[host_preprocess] = feat_idx
                emitted_id_map[feat_idx] = feat_idx
                functions += f'''
void PreprocFeat{feat_idx}(
    const std::vector<std::string> &{vars_.raw_data}, const int {vars_.max_concurrent_blocks},
    int &{vars_.num_tiles_ref}, int &{vars_.num_blocks_ref},
    int &{vars_.output_size_ref}, int &{vars_.extra_buffer_size_ref},
    std::vector<int> *{vars_.data}, int *{vars_.scala_data}) {{
{host_preprocess}
}}

'''

        host = f'''
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>,
           std::vector<int2>, std::vector<int>, std::vector<std::vector<int>>,
           std::vector<int>>
preprocess(const std::vector<std::vector<std::string>> &raw_data,
           const int max_concurrent_blocks) {{
  std::vector<int> task_tiles({num_feats});
  std::vector<int> task_blocks({num_feats});
  typedef void (*PreprocFeatT)(const std::vector<std::string> &, const int,
                               int &, int &, int &, int &, std::vector<int> *,
                               int *);
'''

        host += "  std::vector<PreprocFeatT> preproc_feats = {\n"
        host += ",\n".join([f"      PreprocFeat{emitted_id_map[feat_idx]}"
                            for feat_idx in range(num_feats)]) + "\n"
        host += "  };\n\n"

        data_offsets = [0]
        scala_data_offsets = [0]
        for schedule, params in self.feat_schedules:
            data_cnt, scala_data_cnt = schedule.UsedArgCounts()
            data_offsets.append(data_offsets[-1] + data_cnt)
            scala_data_offsets.append(scala_data_offsets[-1] + scala_data_cnt)

        host += "  std::vector<int> data_offsets = {\n"
        host += ",\n".join([f"      {offset}" for offset in data_offsets]) + "\n"
        host += "  };\n\n"

        host += "  std::vector<int> scala_data_offsets = {\n"
        host += ",\n".join([f"      {offset}" for offset in scala_data_offsets]) + "\n"
        host += "  };\n\n"

        # TODO: output_offsets are useless
        host += f'''
  std::vector<std::vector<int>> data(data_offsets.back());
  std::vector<int> scala_data(scala_data_offsets.back());

  std::vector<int> output_offsets({num_feats + 1}, 0);
  std::vector<int> extra_buffer_offsets({num_feats + 1}, 0);

  // TODO: use thread pool
  std::vector<std::thread> threads({num_feats});
  for (int i = 0; i < threads.size(); ++i) {{
    threads[i] = std::thread(preproc_feats[i],
        raw_data[i], max_concurrent_blocks,
        std::ref(task_tiles[i]), std::ref(task_blocks[i]),
        std::ref(output_offsets[i + 1]), std::ref(extra_buffer_offsets[i + 1]),
        &data[data_offsets[i]], &scala_data[scala_data_offsets[i]]);
  }}

  for (auto &thread : threads)
    thread.join();

  for (int i = 0; i < {num_feats}; ++i) {{
    output_offsets[i + 1] += output_offsets[i];
    extra_buffer_offsets[i + 1] += alignmem(extra_buffer_offsets[i]);
  }}

  std::vector<int> data_buffer_offsets({num_feats + 1}, 0);
  for (int i = 0; i < {num_feats}; ++i) {{
    int buffer_size = 0;
    for (int j = data_offsets[i]; j < data_offsets[i + 1]; ++j) {{
      int n_pad = align_need_pad_elem<int>(data[j].size());
      for (int k = 0; k < n_pad; ++k) {{
        data[j].push_back(0);
      }}
      buffer_size += data[j].size() * sizeof(int);
    }}
    data_buffer_offsets[i + 1] = data_buffer_offsets[i] + buffer_size;
  }}

'''

        if fixed_task_block_nums:
            host += "  std::vector<int2> task_mapping = {\n"
            for feat_id, fixed_task_blocks in enumerate(fixed_task_block_nums):
                for task_bid in range(fixed_task_blocks):
                    host += f"      make_int2({feat_id}, {task_bid}),\n"
            host += "  };\n"
        else:
            host += '''
  std::vector<int2> task_mapping;
  GetTaskMapping(task_blocks, task_mapping);
'''

        host += '''
  // TODO: allocate and return torch::Tensor directly
  return std::make_tuple(data_buffer_offsets, output_offsets, extra_buffer_offsets,
                         task_mapping, task_tiles, data, scala_data);
}
'''

        return functions + host

    def emit_host_op(self) -> str:
        return '''
// Define a PyBind11 compatible conversion for int2
namespace pybind11 {
namespace detail {
  template <> struct type_caster<int2> {
  public:
    PYBIND11_TYPE_CASTER(int2, _("int2"));

    // Conversion part (Python -> C++)
    bool load(handle src, bool) {
      if (!src.is_none()) {
        auto tup = reinterpret_borrow<tuple>(src);
        if (tup.size() != 2)
          return false;

        value.x = tup[0].cast<int>();
        value.y = tup[1].cast<int>();
        return !PyErr_Occurred();
      }
      return false;
    }

    // Conversion part (C++ -> Python)
    static handle cast(const int2& src, return_value_policy, handle) {
      return make_tuple(src.x, src.y).release();
    }
  };
}
}

PYBIND11_MODULE(recom_host, m) {
  m.def("preprocess", &preprocess, "Preprocess the features");
}
'''

    def emit_kernel(self, occupancy: int = None, inline: bool = True, timing: bool = False,
                    timing_branch: bool = False, fixed_task_block_nums: List[int] = None) -> str:
        num_feats = len(self.feat_schedules)
        emitted_code_id_map: Dict[str, int] = {}
        emitted_id_map: Dict[int, int] = {}

        embed_dims: List[int] = [sched.meta.embed_dim for sched, params in self.feat_schedules]
        embed_dim_offsets = [0]
        for embed_dim in embed_dims:
            embed_dim_offsets.append(embed_dim + embed_dim_offsets[-1])
        functions = f"constexpr int EMBED_DIM_SUM = {embed_dim_offsets[-1]};\n"

        temp_storage_decl = "union KnlTempStorage {\n"
        for feat_idx, (schedule, params) in enumerate(self.feat_schedules):
            shmem_type = schedule.EmitShmemType(params)
            schedule_body = schedule.EmitScheduleBody(params)
            key = shmem_type + "\n" + schedule_body
            if key in emitted_code_id_map.keys():
                emitted_id_map[feat_idx] = emitted_code_id_map[key]
            else:
                emitted_code_id_map[key] = feat_idx
                emitted_id_map[feat_idx] = feat_idx
                temp_storage_decl += f"  {shmem_type} f{feat_idx};\n"

                vars_ = ScheduleVars()
                functions += f'''
__device__ {"__forceinline__" if inline else "__noinline__"} void ProcessFeat{feat_idx}(
    const char *__restrict__ {vars_.embed_table},
    const char *__restrict__ {vars_.arg_buffer},
    const int *__restrict__ {vars_.scala_args},
    int {vars_.task_bid}, int {vars_.task_blocks}, int {vars_.num_tiles},
    int {vars_.embed_dim_offset},
    char *__restrict__ {vars_.shmem},
    char *__restrict__ {vars_.output},
    char *__restrict__ {vars_.extra_buffer},
    int *__restrict__ {vars_.task_barriers}) {{
  {schedule_body}
}}

'''

        temp_storage_decl += "};\n\n"

        if inline:
            functions += "__device__ int process_func_ids[] = {\n"
            functions += ",\n".join([f"    {emitted_id_map[feat_idx]}"
                                     for feat_idx in range(num_feats)])
            functions += "\n};\n\n"
        else:
            functions += '''
typedef void (*ProcessFeatT)(const char *, const char *, const int *,
                             int, int, int, int, char *, char *, char *, int *);
__device__ ProcessFeatT process_feats[] = {
'''
            functions += ",\n".join([f"    ProcessFeat{emitted_id_map[feat_idx]}"
                                    for feat_idx in range(num_feats)])
            functions += "\n};\n\n"

        scala_arg_offsets = [0]
        for schedule, params in self.feat_schedules:
            data_cnt, scala_data_cnt = schedule.UsedArgCounts()
            scala_arg_offsets.append(scala_arg_offsets[-1] + scala_data_cnt)

        functions += "__device__ uint d_scala_arg_offsets[] = {\n"
        functions += ",\n".join([f"    {scala_arg_offset}"
                                 for scala_arg_offset in scala_arg_offsets])
        functions += "\n};\n\n"

        functions += "__device__ int d_embed_dim_offsets[] = {\n"
        functions += ",\n".join([f"    {embed_dim_offset}"
                                 for embed_dim_offset in embed_dim_offsets])
        functions += "\n};\n\n"

        if fixed_task_block_nums:
            assert len(fixed_task_block_nums) == num_feats
            functions += "__device__ int d_fixed_task_block_nums[] = {\n"
            functions += ",\n".join([f"    {fixed_task_blocks}"
                                     for fixed_task_blocks in fixed_task_block_nums])
            functions += "\n};\n\n"

        if timing:
            timing_begin = '''
  __syncthreads();
  if (threadIdx.x == 0) {
    d_times[blockIdx.x] = clock();
  }
  __syncthreads();

'''
            timing_end = '''
  __syncthreads();
  if (threadIdx.x == 0) {
    d_times[blockIdx.x] = clock() - d_times[blockIdx.x];
  }

'''

        kernel = f'''
__global__ void __launch_bounds__({self.block_threads}{f", {occupancy}" if occupancy else ''}) FusedKnl(
    const int64_t *__restrict__ d_embed_table_ptrs,
    const char *__restrict__ d_arg_buffers,
    const int *__restrict__ d_arg_buffer_offsets,
    const int *__restrict__ d_scala_args,
    char *__restrict__ d_output_buffers,
    char *__restrict__ d_extra_buffers,
    const int *__restrict__ d_extra_buffer_offsets,
    const int2 *__restrict__ d_task_mapping,
    const int *__restrict__ d_task_tiles,
    int *__restrict__ d_task_barriers,
    {'' if not timing else f"int64_t *__restrict__ d_times,"}
    const int max_concurrent_blocks) {{
  const int2 task = d_task_mapping[blockIdx.x];
  const int feat_idx = task.x;
  int task_bid = task.y;
  const int num_tiles = d_task_tiles[feat_idx];
  const int task_blocks = min(num_tiles, max_concurrent_blocks);
{"  const int fixed_task_blocks = d_fixed_task_block_nums[feat_idx];" if fixed_task_block_nums else ''}
  __shared__ KnlTempStorage s;

'''

        if inline:
            if timing and timing_branch:
                kernel += timing_begin
            kernel += "  const int process_func_id = process_func_ids[feat_idx];\n"
            # TODO: potential bug for fixed task blocks if inter-block barrier used
            call_processes = [f'''if (process_func_id == {func_id}) {{
{timing_begin if timing and not timing_branch else ''}
{"    for (; task_bid < task_blocks; task_bid += fixed_task_blocks)" if fixed_task_block_nums else ''}
    ProcessFeat{func_id}(
        reinterpret_cast<const char *>(d_embed_table_ptrs[feat_idx]),
        d_arg_buffers + d_arg_buffer_offsets[feat_idx],
        d_scala_args + d_scala_arg_offsets[feat_idx],
        task_bid, task_blocks, num_tiles,
        d_embed_dim_offsets[feat_idx],
        reinterpret_cast<char *>(&s),
        d_output_buffers,
        d_extra_buffers + d_extra_buffer_offsets[feat_idx],
        d_task_barriers);
{timing_end if timing and not timing_branch else ''}
  }}'''
            for func_id in emitted_code_id_map.values()]
            kernel += "  " + " else ".join(call_processes)
            if timing and timing_branch:
                kernel += timing_end
            kernel += "\n}\n\n"
        else:
            kernel += f'''
{timing_begin if timing else ''}
{"  for (; task_bid < task_blocks; task_bid += fixed_task_blocks)" if fixed_task_block_nums else ''}
  process_feats[feat_idx](
      reinterpret_cast<const char *>(d_embed_table_ptrs[feat_idx]),
      d_arg_buffers + d_arg_buffer_offsets[feat_idx],
      d_scala_args + d_scala_arg_offsets[feat_idx],
      task_bid, task_blocks, num_tiles,
      d_embed_dim_offsets[feat_idx],
      reinterpret_cast<char *>(&s),
      d_output_buffers,
      d_extra_buffers + d_extra_buffer_offsets[feat_idx],
      d_task_barriers);
{timing_end if timing else ''}
}}
'''

        return temp_storage_decl + functions + kernel

    def emit_process_func(self, timing: bool = False, occupancy: int = None) -> str:
        time_tensor_alloc = '''
  auto time_tensor_opts = 
    torch::TensorOptions()
      .dtype(torch::kInt64)
      .device(d_arg_buffers.device());
  auto d_times = torch::empty({total_blocks}, time_tensor_opts);
'''

        time_conversion = '''
  int clock_rate_khz;
  CubDebugExit(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
  d_times = d_times.to(torch::kFloat64);
  d_times = d_times.divide(clock_rate_khz / 1000);
'''

        return f'''
{'torch::Tensor' if not timing else 'std::tuple<torch::Tensor, torch::Tensor>'} process(
    const torch::Tensor &d_embed_table_ptrs,
    const torch::Tensor &d_arg_buffers,
    const torch::Tensor &d_arg_buffer_offsets,
    const torch::Tensor &d_scala_args,
    torch::Tensor &d_extra_buffers,
    const torch::Tensor &d_extra_buffer_offsets,
    const torch::Tensor &d_task_mapping,
    const torch::Tensor &d_task_tiles,
    torch::Tensor &d_task_barriers,
    const int64_t output_size,
    const int64_t max_concurrent_blocks) {{
  auto output_opts = 
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(d_arg_buffers.device());
  const int64_t output_ele_num = output_size / sizeof(float);
  auto d_output_buffer = torch::empty({{output_ele_num}}, output_opts);
  const int total_blocks = d_task_mapping.size(0);
{'' if not timing else time_tensor_alloc}
  size_t dyn_smem_size = {f'get_dyn_smem_size(FusedKnl, {occupancy}, {self.block_threads})' if occupancy else '0'};
  FusedKnl<<<total_blocks, {self.block_threads}, dyn_smem_size>>>(
    d_embed_table_ptrs.data_ptr<int64_t>(),
    reinterpret_cast<const char *>(d_arg_buffers.data_ptr<int>()),
    d_arg_buffer_offsets.data_ptr<int>(),
    d_scala_args.data_ptr<int>(),
    reinterpret_cast<char *>(d_output_buffer.data_ptr<float>()),
    reinterpret_cast<char *>(d_extra_buffers.data_ptr<int8_t>()),
    d_extra_buffer_offsets.data_ptr<int>(),
    reinterpret_cast<const int2 *>(d_task_mapping.data_ptr<int>()),
    d_task_tiles.data_ptr<int>(),
    d_task_barriers.data_ptr<int>(),
    {'' if not timing else 'd_times.data_ptr<int64_t>(),'}
    max_concurrent_blocks
  );

{'' if not timing else time_conversion}

  return {'d_output_buffer' if not timing else 'std::make_tuple(d_output_buffer, d_times)'};
}}
'''

    def emit_get_concurrent_blocks_func(self, occupancy: int = None) -> str:
        return f'''
int64_t get_max_concurrent_blocks_wrapper() {{
  size_t dyn_smem_size = {f'get_dyn_smem_size(FusedKnl, {occupancy}, {self.block_threads})' if occupancy else '0'};
  return get_max_concurrent_blocks(FusedKnl, {self.block_threads}, dyn_smem_size);
}}
'''

    def emit_process_ops(self) -> str:
        return '''
TORCH_LIBRARY(recom, m) {
  m.def("get_gpu_pointers_array", get_gpu_pointers_array);
  m.def("get_max_concurrent_blocks", get_max_concurrent_blocks_wrapper);
  m.def("process", process);
}
'''

    def emit_if_updated(self, fname: str, code: str) -> bool:
        if os.path.exists(fname):
            with open(fname) as f:
                old = f.read()
                if old.strip() == code.strip():
                    return False

        with open(fname, "w") as f:
            f.write(code)
        return True

    def emit(self, ofname: str, occupancy: int = None, inline: bool = True,
             tune: bool = False, timing: bool = False, timing_branch: bool = False,
             fixed_task_block_nums: List[int] = None) -> None:
        code = ""
        code += self.emit_headers()
        code += self.emit_kernel(timing=timing, timing_branch=timing_branch,
                                 occupancy=occupancy, inline=inline,
                                 fixed_task_block_nums=fixed_task_block_nums)
        code += self.emit_process_func(timing=timing, occupancy=occupancy if tune else None)
        code += self.emit_get_concurrent_blocks_func(occupancy=occupancy if tune else None)
        code += self.emit_process_ops()

        self.emit_if_updated(ofname, code)

    def emit_host(self, ofname: str, fixed_task_block_nums: List[int] = None) -> None:
        code = ""
        code += self.emit_host_op_headers()
        code += self.emit_host_preprocess(fixed_task_block_nums=fixed_task_block_nums)
        code += self.emit_host_op()

        self.emit_if_updated(ofname, code)

    def emit_cmake_list(self, ofname: str, code_fname: str, host_fname: str,
                        subdirs: List[str] = [], includes: List[str] = [],
                        links: List[str] = [], host_links: List[str] = [],
                        extra: str = "", debug: bool = False) -> None:
        basic_builder = BasicBuilder()
        basic_lib = basic_builder.build()
        includes = includes + [basic_builder.include]
        links = links + [basic_lib]
        host_links = host_links + [basic_lib]

        add_subdirs = '\n'.join([f"add_subdirectory({subdir})" for subdir in subdirs])
        add_includes = '\n'.join([f"include_directories({include})" for include in includes])
        add_links = '\n'.join([f"target_link_libraries(recom PUBLIC {link})" for link in links])
        add_host_links = '\n'.join([f"target_link_libraries(recom_host PUBLIC {link})" for link in host_links])

        recom_host = f'''
execute_process(COMMAND "${{Python_EXECUTABLE}}" -c "import os; import pybind11; print(os.path.dirname(pybind11.__file__))" OUTPUT_VARIABLE PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH ${{PYBIND11_DIR}})
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(recom_host {host_fname})
target_compile_features(recom_host PUBLIC cxx_std_17)
{add_host_links}
'''

        debug_flags = f'''
target_compile_options(recom PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -g;
                        -G;
                        -O0;
                       >)
'''

        recom = f'''
add_library(recom SHARED {code_fname})
target_compile_features(recom PUBLIC cxx_std_17)

target_link_libraries(recom PUBLIC ${{Python_LIBRARIES}})
{add_links}

execute_process(COMMAND "${{Python_EXECUTABLE}}" -c "import os; import torch; print(os.path.dirname(torch.__file__))" OUTPUT_VARIABLE TORCH_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH ${{TORCH_DIR}})
find_package(Torch REQUIRED)
target_link_libraries(recom PUBLIC ${{TORCH_LIBRARIES}})

# debug flags
{debug_flags if debug else ''}
'''

        code = f'''
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(recom_project)

# Add subdirectories
{add_subdirs}

# Add include directories
{add_includes}

find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${{Python_INCLUDE_DIRS}})

# Add target recom_host
{recom_host if host_fname else ''}

# Add target recom
{recom if code_fname else ''}

# Extra area
{extra}
'''

        self.emit_if_updated(ofname, code)
