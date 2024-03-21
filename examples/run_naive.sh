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


m=$1
# mkdir -p compress
# python compress.py -m models/$m -o compress/$m
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tune.py -m compress/$m -o compress/$m/output -n 48 --local_only --naive
python decompress.py -i compress/$m -o models/$m --naive
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tune.py -m models/$m -o models/$m/output -n 48 --naive
