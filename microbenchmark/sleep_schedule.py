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

# Use this to prevent merging schedules
class SleepParams(NamedTuple):
    ident: int

class SleepSchedule(ScheduleBase):
    def EmitScheduleBody(self, params: SleepParams) -> str:
        vars_ = self.sched_vars
        schedule_body = f'''
  // Identity: {params.ident}
  int sleep_time_in_us = {vars_.scala_args}[0];
  __nanosleep(sleep_time_in_us * 1000);
'''
        return schedule_body

    def EmitShmemType(self, params: SleepParams) -> str:
        return "int"

    def EmitHostPreprocess(self, params: SleepParams) -> str:
        raise NotImplementedError

    def UsedArgCounts(self) -> Tuple[int, int]:
        # scala_data[1] = sleep_time_in_ns
        return 0, 1

    def EmitHeaders(self) -> List[str]:
        return []

    def EmitPrepropHeaders(self) -> List[str]:
        raise NotImplementedError

    def GenParamCandidates(self, stats: InputStats = None) -> List[SleepParams]:
        raise NotImplementedError
