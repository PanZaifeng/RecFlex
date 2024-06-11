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


pip install tensorflow==2.6.2
pip install protobuf==3.20.3

apt-get install -y git 
apt-get install -y libgmp-dev

curl https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazelisk/v1.18.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

export TF_NEED_CUDA=1
export TF_CUDA_VERSION=11
export TF_CUDNN_VERSION=8
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export TF_CONFIGURE_IOS=False
export TF_SET_ANDROID_WORKSPACE=False
export TF_NEED_ROCM=0
export TF_CUDA_CLANG=0
export TF_NEED_TENSORRT=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export CC_OPT_FLAGS=-Wno-sign-compare
export PYTHON_BIN_PATH=/usr/bin/python
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_CUDA_COMPUTE_CAPABILITIES=7.5,8.0,8.6

