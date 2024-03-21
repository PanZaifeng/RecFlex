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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--table_config_path", type=str, required=True)
    args = parser.parse_args()

    embed_dims = []
    with open(args.table_config_path) as table_config:
        for line in table_config.readlines():
            row_num, embed_dim = line.split(",")
            embed_dims.append(int(embed_dim))

    counts, bins = np.histogram(embed_dims, np.arange(0, 130, 4))

    figsize = (5, 2.6)
    fontsize = 16

    fig = plt.figure(figsize=figsize)
    plt.hist(
        bins[:-1], bins, weights=counts, color="#9DC3E6", edgecolor="k", linewidth=1
    )

    # plt.gca().set_xscale("log")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.xlabel("Embedding Dimension", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.tight_layout()

    fig.savefig(f"embed_dim_frequency.pdf", bbox_inches="tight")
