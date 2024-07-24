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

script_path=$(realpath "$0")
artifact_dir=$(dirname "$0")
recflex_dir=$(realpath ${artifact_dir}/..)

docker build -f ./Dockerfile -t recflex:latest .
docker run --rm -v ${recflex_dir}:/recflex --gpus all -it recflex:latest sh -c \
    "pip install /recflex && /recflex/post_install.sh \
        && /recflex/SC_artifact/datagen.sh \
        && /recflex/SC_artifact/run_recflex.sh \
        && /recflex/SC_artifact/run_torchrec.sh"

docker build -f ./TF.Dockerfile -t recom:latest .
docker run --rm -v ${recflex_dir}:/recflex --gpus all -it recom:latest sh -c \
    "pip install /recflex && /recflex/SC_artifact/run_tf_recom.sh"

docker run --rm -v ${recflex_dir}:/recflex --gpus all \
    -it nvcr.io/nvidia/merlin/merlin-hugectr:23.04 sh -c \
    "cd /recflex && python setup.py install && /recflex/SC_artifact/run_hugectr.sh"

docker run --rm -v ${recflex_dir}:/recflex --gpus all -it recflex:latest \
    /recflex/SC_artifact/plot.sh
