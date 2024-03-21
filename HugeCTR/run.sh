#!/bin/bash
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


CUDA_VISIBLE_DEVICES=0 python prof_kernel.py -b ../examples/models/ -m D E -o output && python prof_kernel.py -b ../examples/models/ -m D -o output -g
CUDA_VISIBLE_DEVICES=1,2 python prof_kernel.py -b ../examples/models/ -m D E -o output --e2e &
CUDA_VISIBLE_DEVICES=3 python prof_kernel.py -b ../examples/models/ -m D -o output --e2e -g &
wait
