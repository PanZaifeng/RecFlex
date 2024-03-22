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

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from typing import List


def parse_naive_results(dirname: str) -> List[float]:
    results = []
    sid = 0
    while True:
        result_path = os.path.join(dirname, f"s{sid}", "result.txt")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(float(f.read().strip()))
        else:
            break
        sid += 1
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--naive_directories", nargs="+", type=str, required=True)
    args = parser.parse_args()

    data_list: List[List[float]] = []
    for dirname in args.naive_directories:
        latency = parse_naive_results(dirname)
        perf = 1 / np.array(latency)
        data = perf / np.max(perf)
        print(1 - np.min(data))
        data_list.append(data)

    figsize = (9, 3.2)
    fontsize = 18
    lengen_fontsize = 16

    fig = plt.figure(figsize=figsize)

    for i, data in enumerate(data_list):
        plt.plot(data, label=f"Feature {i}")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.xlabel("Schedule ID", fontsize=fontsize)
    plt.ylabel("Normalized\nPerformance", fontsize=fontsize)
    plt.legend(fontsize=lengen_fontsize, loc="upper left")
    plt.tight_layout()

    fig.savefig("opportunity.pdf", bbox_inches="tight")
