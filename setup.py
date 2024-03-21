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
from setuptools import setup, find_packages

work_directory = os.path.abspath(os.path.dirname(__file__))

setup(
    name="RecFlex",
    version="0.0.1",
    packages=find_packages(),
    package_data={
        "RecFlex": [
            f"{work_directory}/RecFlex/sched_basic_csrc/*",
            f"{work_directory}/RecFlex/sched_basic_csrc/include/*",
        ]
    },
    install_requires=[
        "scipy",
        "pybind11"
    ],
    description="RecFlex Package",
    author="RecFlex Authors",
    license='Apache License 2.0',
)
