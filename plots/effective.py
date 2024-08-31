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

plt.rcParams["pdf.fonttype"] = 42


def plot_mutation(data, select, color, label):
    plt.plot(data, label=label, color=color)
    plt.scatter(select, data[select], edgecolor=color, facecolors='none',
                marker='o', s=100)


def plot(data_list, selects, save):
    figsize = (9, 4.25)
    fontsize = 18 

    fig = plt.figure(figsize=figsize)

    colors = ["blue", "green", "red", "yellow"]
    for i, (data, select, color) in enumerate(zip(
        data_list, selects, colors
    )):
        plot_mutation(data, select, color, f"Feature {i}")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.legend(bbox_to_anchor=(0.47, 1.05), loc="lower center", ncol=3,
               fontsize=fontsize - 2, handletextpad=0.5, columnspacing=4.0,
               frameon=False)
    plt.xlabel("Selected Schedule ID", fontsize=fontsize, labelpad=10)
    plt.ylabel("Normalized Kernel\nPerformance", fontsize=fontsize, labelpad=10)
    plt.tight_layout()

    fig.savefig(save, bbox_inches="tight")


def read_data(fname):
    with open(fname) as f:
        latency = list(map(float, f.read().strip().split('\n')))
        perf = 1 / np.array(latency)
        data = perf / np.max(perf)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation_results", nargs="+", type=str, required=True)
    parser.add_argument("--selects", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=str, default="effective.pdf")
    args = parser.parse_args()

    data_list = []
    for fname in args.mutation_results:
        data = read_data(fname)
        data_list.append(data)

    plot(data_list, args.selects, args.output)
