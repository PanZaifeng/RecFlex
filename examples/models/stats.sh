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
result_dir=${cur_dir}/results
rm -f ${result_dir}/result.txt
for m in A B C D E; do
  mdir=${cur_dir}/$m
  nsys stats $mdir/output/optimal/report.nsys-rep | grep Fused | awk -v m="$m" '{sum+=$2} END {print m","sum/1e6}' >> ${result_dir}/result.txt
done
