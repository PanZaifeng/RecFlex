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
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["pdf.fonttype"] = 42


def plot_bars(data_list, fontsize=22, save="noinline.pdf"):
    models = ["A", "B", "C", "D", "E"]
    labels = ["Jump Table", "If-else Branches"]
    colors = ["#DCF8E6", "#F8CBAD"]
    hatches = ["/", "-"]

    fig = plt.figure(figsize=(8, 4.5))

    width = 1.0 / (len(data_list) + 1)
    location = np.arange(len(models))

    for i, (data, label, color, hatch) in enumerate(zip(
        data_list, labels, colors, hatches
    )):
        plt.bar(location + width * i, data, width=width, label=label,
                color=color, hatch=hatch, edgecolor="k", alpha=1)
    plt.bar(location + width * len(labels) / 2,
            np.zeros_like(data), tick_label=models)
    
    plt.legend(bbox_to_anchor=(0.45, 1.05), loc="lower center", ncol=2,
               fontsize=fontsize - 2, handletextpad=0.5, columnspacing=2.0,
               frameon=False)
    plt.xlabel("Models", fontsize=fontsize, labelpad=5)
    plt.ylabel("Normalized Kernel\nPerformance", fontsize=fontsize, labelpad=10)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    fig.savefig(save, bbox_inches="tight")


def read_data(fname):
    results = {}
    with open(fname) as f:
        for line in f.readlines():
            model, time_str = line.strip().split(',')
            t = float(time_str)
            results[model] = t

    data = []
    models = ["A", "B", "C", "D", "E"]
    for model in models:
        if model not in results.keys():
            data.append(np.inf)
        else:
            data.append(results[model])
    return data


def normalize(data_list):
    norm = []
    recflex = np.array(data_list[-1])
    for data in data_list:
        norm.append(recflex / np.array(data))
    return norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recflex", type=str, required=True)
    parser.add_argument("--noinline", type=str, required=True)
    parser.add_argument("--output", type=str, default="noinline.pdf")
    args = parser.parse_args()

    data_list = []
    for fname in [args.noinline, args.recflex]:
        data = read_data(fname)
        data_list.append(data)

    data_list = normalize(data_list)
    print([np.mean(data) for data in data_list])
    print([np.mean(1 / data) for data in data_list])
    plot_bars(data_list, save=args.output)
