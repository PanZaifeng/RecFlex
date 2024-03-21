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

from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Tuple


class InputStats(NamedTuple):
    mean: float
    var: float
    maximum: float
    minmum: float


class MetaData(NamedTuple):
    embed_dim: int
    data_type: str = "float"


class ScheduleVars(NamedTuple):
    embed_table: str = "d_embed_table"
    shmem: str = "s"
    task_bid: str = "task_bid"
    task_blocks: str = "task_blocks"
    num_tiles: str = "num_tiles"
    task_barriers: str = "d_task_barriers"
    arg_buffer: str = "d_arg_buffer"
    scala_args: str = "d_scala_args"
    output: str = "d_output"
    extra_buffer: str = "d_extra_buffer"
    embed_dim_offset: str = "embed_dim_offset"
    EMBED_DIM_SUM: str = "EMBED_DIM_SUM"


class PreprocessVars(NamedTuple):
    raw_data: str = "raw_data"
    data: str = "data"
    scala_data: str = "scala_data"
    max_concurrent_blocks: str = "max_concurrent_blocks"
    num_tiles_ref: str = "num_tiles_ref"
    num_blocks_ref: str = "num_blocks_ref"
    output_size_ref: str = "output_size_ref"
    extra_buffer_size_ref: str = "extra_buffer_size_ref"


class ScheduleBase(ABC):
    def __init__(self, meta_data: MetaData, block_threads: int):
        self.meta: MetaData = meta_data
        self.block_threads = block_threads
        self.sched_vars: ScheduleVars = ScheduleVars()
        self.pre_vars: PreprocessVars = PreprocessVars()

    @abstractmethod
    def EmitScheduleBody(self, params: NamedTuple) -> str:
        pass

    @abstractmethod
    def EmitShmemType(self, params: NamedTuple) -> str:
        pass

    @abstractmethod
    def UsedArgCounts(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def EmitHostPreprocess(self, params: NamedTuple) -> str:
        pass

    @abstractmethod
    def EmitHeaders(self) -> List[str]:
        pass

    @abstractmethod
    def EmitPrepropHeaders(self) -> List[str]:
        pass

    @abstractmethod
    def GenParamCandidates(self, stats: InputStats = None) -> List:
        pass