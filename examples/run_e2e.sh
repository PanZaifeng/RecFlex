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


outdir=end_to_end
mkdir -p $outdir
for m in A B C D E; do
  CUDA_VISIBLE_DEVICES=0 nsys profile -c cudaProfilerApi -t cuda -f true -o $outdir/$m python end_to_end.py -t models/$m/table_config.txt -b models/$m/output/optimal/build -d models/$m/data
  nsys stats --force-export true -f csv --report gpukernsum $outdir/$m.nsys-rep | awk -v m="$m" -F, 'BEGIN {tsum=0} {if(NR>=2) tsum+=$2} END {print m","tsum/1e6}' >> $outdir/result.txt
done
