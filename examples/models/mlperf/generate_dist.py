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
import numpy as np

def generate(num_batches, ofile):
    # quantile used by MLPerf
    quantile = np.array([100, 100, 200, 200, 200, 200, 200, 200, 200, 200,
                         200, 200, 200, 200, 300, 300, 400, 500, 600, 700],
                        dtype=np.int32)

    l = len(quantile)
    with open(ofile, "w") as f:
        for i in range(num_batches):
            pr = np.random.randint(low=0, high=l)
            qs = quantile[pr]
            f.write(str(qs) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--num_batches", type=int, default=128)
    parser.add_argument("-o", "--ofile", type=str, required=True)
    args = parser.parse_args()
    generate(args.num_batches, args.ofile)
