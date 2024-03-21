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


cur_dir=$(dirname "$0")
for m in A B C D; do
  mdir=${cur_dir}/$m
  python ${cur_dir}/../../data_synthesis/data_generate.py -c $mdir/data_config.txt -t $mdir/table_config.txt -n 32 -o $mdir/data
done

ln -s -T $(realpath $cur_dir/D/data) $cur_dir/E/data
