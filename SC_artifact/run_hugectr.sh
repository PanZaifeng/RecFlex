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

artifact_dir=$(realpath $(dirname "$0"))
recflex_dir=$artifact_dir/..
hugectr_dir=$recflex_dir/HugeCTR
models_dir=$recflex_dir/examples/models
cd $hugectr_dir

outdir=outputs
mkdir -p $outdir
CUDA_VISIBLE_DEVICES=0 python prof_kernel.py \
    -b $models_dir -m D E -o $outdir

resdir=$artifact_dir/results
mkdir -p $resdir
rm -f $resdir/hugectr_{kern,e2e}.txt
for m in D E; do
  nsys stats --force-export true -f csv --report gpukernsum $outdir/${m}_c.nsys-rep \
      | grep embedding \
      | awk -v m="$m" -F, 'BEGIN {tsum=0} {tsum+=$2} END {print m","tsum/1e6}' \
      >> $resdir/hugectr_kern.txt
  nsys stats --force-export true -f csv --report gpukernsum $outdir/${m}_c.nsys-rep \
      | awk -v m="$m" -F, 'BEGIN {tsum=0} {if(NR>=2) tsum+=$2} END {print m","tsum/1e6}' \
      >> $resdir/hugectr_e2e.txt
done