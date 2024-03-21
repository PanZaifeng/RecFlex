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

import os
import numpy as np
from typing import Dict, List, Tuple

from RecFlex.code_emitter import ScheduleBase


def parse_schedule_types(config_fname: str, schedule_mapping: Dict[str, ScheduleBase],
                         sep: str = ',') -> List[List[ScheduleBase]]:
    feat_schedule_types = []
    with open(config_fname) as f:
        for line in f.readlines():
            schedule_types = list(map(schedule_mapping.get, line.strip().split(sep)))
            feat_schedule_types.append(schedule_types)
    return feat_schedule_types

def parse_raw_data_batches(batch_sizes_fname: str, data_fnames: List[str], sample_batch_nums: int = 128,
                           seed: int = 0, feat_major: bool = True) -> List[List[List[str]]]:
    '''
    Return: (num_feats, sample_batch_nums, batch_size) (feat_major=True)
            (sample_batch_nums, num_feats, batch_size) (feat_major=False)
    '''
    np.random.seed(seed)

    with open(batch_sizes_fname) as f:
        batch_sizes = [int(line.strip()) for line in f.readlines()]
    line_offsets = np.cumsum([0] + batch_sizes)
    sample_batch_nums = min([sample_batch_nums, len(batch_sizes)])
    sample_batch_ids = np.sort(np.random.choice(len(batch_sizes), sample_batch_nums, replace=False))

    num_feats = len(data_fnames)
    if feat_major:
        raw_data_batches = [[] for _ in range(num_feats)]
        for feat_id, fname in enumerate(data_fnames):
            with open(fname) as f:
                lines = f.read().splitlines()
                for batch_id in sample_batch_ids:
                    start = line_offsets[batch_id]
                    end = line_offsets[batch_id + 1]
                    raw_data_batches[feat_id].append(lines[start:end])
    else:
        raw_data_batches = [[] for _ in range(sample_batch_nums)]
        for feat_id, fname in enumerate(data_fnames):
            with open(fname) as f:
                lines = f.read().splitlines()
                for i, batch_id in enumerate(sample_batch_ids):
                    start = line_offsets[batch_id]
                    end = line_offsets[batch_id + 1]
                    raw_data_batches[i].append(lines[start:end])
    
    return raw_data_batches

def parse_table_shapes(config_fname: str, sep: str = ',') -> List[Tuple[int, int]]:
    table_shapes = []
    with open(config_fname) as f:
        for line in f.readlines():
            table_shapes.append(tuple(map(int, line.strip().split(sep))))
    return table_shapes

def parse_knl_total_time(knl: str, nsys_report: str, reptype: str = "gpukernsum"):
    cmd = (
        f"nsys stats --force-export -f csv --report {reptype} {nsys_report} "
        f"| grep '{knl}' "
        f"| awk -F , 'BEGIN {{tsum = 0}} {{tsum += $2}} END {{print tsum}}'"
    )
    return int(os.popen(cmd).read().strip())

def parse_first_knl_total_time(nsys_report: str, reptype: str = "gpukernsum"):
    cmd = (
        f"nsys stats --force-export -f csv --report {reptype} {nsys_report} "
        f"| grep -A 1 'Total Time' "
        f"| awk -F , '{{if (NR == 2) print $2}}'"
    )
    return int(os.popen(cmd).read().strip())
