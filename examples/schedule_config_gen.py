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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--data_config_path", type=str, required=True)
    parser.add_argument("-s", "--schedule_config_path", type=str, required=True)
    args = parser.parse_args()

    schedule_configs = []
    with open(args.data_config_path) as config_file:
        for line in config_file.readlines():
            feat_config = line.split(",")
            feat_type, feat_gen_num = feat_config[:2]
            feat_config = list(
                map(lambda x: int(x) if "." not in x else float(x), feat_config[2:])
            )

            feat_gen_num = int(feat_gen_num)
            if feat_type == "one-hot":
                schedule_configs.extend(["OneHotSchedule"] * feat_gen_num)
            elif feat_type in ["multi-hot", "multi-hot-static", "multi-hot-one-side"]:
                schedule_configs.extend([','.join([
                    "ReduceByKeySchedule",
                    # "MultiBlockPerSampleSchedule",
                    "OneBlockMultiSampleSchedule",
                    "WarpPerSampleSchedule",
                ])] * feat_gen_num)

    with open(args.schedule_config_path, "w") as sched_config_file:
        for schedule_config in schedule_configs:
            sched_config_file.write(f"{schedule_config}\n")
