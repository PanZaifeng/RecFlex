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

from RecFlex.code_emitter import ScheduleBase, InputStats

from typing import List, NamedTuple, Tuple

class FenceParams(NamedTuple):
    ident: int

class FenceSchedule(ScheduleBase):
    def EmitScheduleBody(self, params: FenceParams) -> str:
        vars_ = self.sched_vars
        schedule_body = f'''
  for (int i = 0; i < {params.ident}; ++i) {{
    __threadfence();
  }}
'''
        return schedule_body

    def EmitShmemType(self, params: FenceParams) -> str:
        return "int"

    def EmitHostPreprocess(self, params: FenceParams) -> str:
        raise NotImplementedError

    def UsedArgCounts(self) -> Tuple[int, int]:
        return 0, 0

    def EmitHeaders(self) -> List[str]:
        return []

    def EmitPrepropHeaders(self) -> List[str]:
        raise NotImplementedError

    def GenParamCandidates(self, stats: InputStats = None) -> List[FenceParams]:
        raise NotImplementedError
