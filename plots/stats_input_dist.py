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

from matplotlib import pyplot as plt
import numpy as np
import argparse

plt.rcParams["pdf.fonttype"] = 42


def split_data_line(data_line):
    data_line = data_line.strip()
    if data_line == "":
        return []
    return list(map(int, data_line.split(",")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_files", nargs="+", type=str, required=True)
    args = parser.parse_args()

    data = []
    N = 50
    for data_file in args.data_files:
        with open(data_file) as df:
            feat_data = []
            for i, line in enumerate(df.readlines()):
                if i == N:
                    break
                feat_data.append(split_data_line(line))
            data.append(feat_data)

    figsize = (8, 4)
    fontsize = 22
    lengen_fontsize = 16

    fig = plt.figure(figsize=figsize)

    sample_lengths_list = [np.array([len(sample) for sample in feat_data]) for feat_data in data]
    for i, sample_lengths in enumerate(sample_lengths_list):
        plt.plot(sample_lengths, label=f"Feature {i}")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.xlabel("Sample ID", fontsize=fontsize)
    plt.ylabel("Pooling Factor", fontsize=fontsize)
    plt.legend(fontsize=lengen_fontsize, loc="upper right")
    plt.tight_layout()

    fig.savefig(f"sample_pooling.pdf", bbox_inches="tight")
