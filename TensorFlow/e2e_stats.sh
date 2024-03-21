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


outdir=output
for x in tf recom; do
  for m in A B C D E; do
    nsys stats -f csv --report gpukernsum $outdir/$x/$m.nsys-rep | awk -v m="$m" -F, 'BEGIN {tsum=0} {if(NR>=2) tsum+=$2} END {print m","tsum/1e6}' >> $outdir/$x/result_e2e.txt
  done
done
