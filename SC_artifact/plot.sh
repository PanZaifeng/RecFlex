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
plots_dir=$recflex_dir/plots
resdir=$artifact_dir/results

outdir=$artifact_dir/outputs
mkdir -p $outdir
for x in kern e2e; do
  python $plots_dir/$x.py \
      --output $outdir/$x.pdf \
      --recflex $resdir/recflex_$x.txt \
      --tf $resdir/tf_$x.txt \
      --recom $resdir/recom_$x.txt \
      --torchrec $resdir/torchrec_$x.txt \
      --hugectr $resdir/hugectr_$x.txt
done