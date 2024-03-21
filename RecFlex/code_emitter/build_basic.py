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
import RecFlex

from RecFlex.utils import compile_build

class BasicBuilder:
    def __init__(self):
        self.recflex_dir = os.path.dirname(RecFlex.__file__)
        self.csrc_dir = f"{self.recflex_dir}/sched_basic_csrc"
        self.include = f"{self.csrc_dir}/include"

    def build(self, dirname: str = "build") -> str:
        build_dir = f"{self.csrc_dir}/{dirname}"
        lib = f"{build_dir}/libsched_basic.so"
        if not os.path.exists(lib):
            compile_build(build_dir)
        return lib
