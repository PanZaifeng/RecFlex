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
import fnmatch
import importlib.util

from typing import List
from types import ModuleType


def os_check(command):
    status = os.system(command)
    if status != 0:
        raise Exception(f"Process exit status {status}, command: {command}")


def compile_build(build_dir: str, make_dir: bool = True):
    cwd = os.getcwd()

    print(f"Building {build_dir}")
    if make_dir:
        os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)
    os_check("cmake .. && make")

    os.chdir(cwd)


def find_file(directory: str, pattern: str) -> str:
    for root, _, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            return os.path.join(root, filename)
    raise Exception(f"Cannot find {pattern} under {directory}")


def import_from_path(module_name: str, module_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_data_fnames(data_dir: str) -> List[str]:
    data_fnames: List[str] = []
    feat_id = 0
    while True:
        fname = os.path.join(data_dir, f"f{feat_id}.txt")
        if os.path.exists(fname):
            data_fnames.append(fname)
            feat_id += 1
        else:
            break
    return data_fnames
