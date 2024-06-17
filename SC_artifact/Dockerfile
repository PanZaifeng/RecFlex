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

# install cmake
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz -o /tmp/cmake-3.28.3-linux-x86_64.tar.gz
RUN tar -zxvf /tmp/cmake-3.28.3-linux-x86_64.tar.gz -C /usr/local && rm /tmp/cmake-3.28.3-linux-x86_64.tar.gz
ENV PATH=/usr/local/cmake-3.28.3-linux-x86_64/bin:$PATH

# install torch and torchrec
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install fbgemm-gpu==0.6.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchmetrics==1.0.3
RUN pip install torchrec==0.6.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install matplotlib==3.3.4