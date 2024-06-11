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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_npz", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=51200)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    for fid in range(26):
        feature_data = data[str(fid)]
        with open(f"{args.output_dir}/f{fid}.txt", "w") as f:
            count = 0
            while count < args.num_samples:
                for row in feature_data:
                    s = ",".join(map(str, row))
                    f.write(s + "\n")
                    count += 1
                    if count == args.num_samples:
                        break

