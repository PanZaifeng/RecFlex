FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update

# install python3.8 and pip
RUN apt-get install -y curl python3.8 python3.8-dev python3.8-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# install nsys
RUN curl -L https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb \
    -o /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb \
    && rm /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb

# install tensorflow
RUN pip install tensorflow==2.6.2
RUN pip install protobuf==3.20

# install recom dependency
RUN apt-get install -y libgmp-dev
RUN curl https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazelisk/v1.18.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

# set recom environments
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=11 \
    TF_CUDNN_VERSION=8 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
ENV TF_CONFIGURE_IOS=False \
    TF_SET_ANDROID_WORKSPACE=False \
    TF_NEED_ROCM=0 \
    TF_CUDA_CLANG=0 \
    TF_NEED_TENSORRT=0 \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
    CC_OPT_FLAGS=-Wno-sign-compare \
    PYTHON_BIN_PATH=/usr/bin/python \
    USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5,8.0,8.6,9.0
