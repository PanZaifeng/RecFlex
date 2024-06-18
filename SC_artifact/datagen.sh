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

artifact_dir=$(dirname "$0")
recflex_dir=$artifact_dir/..
models_dir=$recflex_dir/examples/models

models=(A B C D)
seeds_tune=(1000 2000 3000 4000)
seeds_test=(5000 6000 7000 8000)
n=${#models[@]}

for ((i=0; i<$n; i++)); do
  m=${models[$i]}
  mdir=${models_dir}/$m
  seed_tune=${seeds_tune[$i]}
  seed_test=${seeds_test[$i]}
  python ${recflex_dir}/data_synthesis/data_generate.py \
      -c $mdir/data_config.txt -t $mdir/table_config.txt -n 32 \
      -s $seed_tune -o $mdir/data
  python ${recflex_dir}/data_synthesis/data_generate.py \
      -c $mdir/data_config.txt -t $mdir/table_config.txt -n 32 \
      -s $seed_test -o $mdir/data_test
done

for f in data data_test; do
  if [ ! -e "$models_dir/E/$f" ]; then
    ln -s -T $(realpath $models_dir/D/$f) $models_dir/E/$f
  fi
done
