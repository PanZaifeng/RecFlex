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


RECOM_CACHE_DIR=RECOM CUDA_VISIBLE_DEVICES=3,4,5,6,7 python prof_kernel.py -b ../examples/models -m A B C D E -o output/recom -l recom/bazel-bin/tensorflow_addons/librecom.so
