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

import ctypes
import os

from typing import List, Tuple

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39


class CudaInfoQuerier:
    def __init__(self):
        self.cuda = ctypes.CDLL("libcuda.so")
        self.gpu_ids: List[int] = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
        self.cuda_init()

    def cuda_check(self, result):
        error_str = ctypes.c_char_p()
        if result != CUDA_SUCCESS:
            self.cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise Exception(
                f"CUDA function call failed with error code {result}: {error_str.value.decode()}"
            )

    def cuda_init(self):
        self.cuda_check(self.cuda.cuInit(0))

    def get_compute_capability(self, dev_ordinal: int = 0) -> Tuple[int, int]:
        device = ctypes.c_int()
        self.cuda_check(self.cuda.cuDeviceGet(ctypes.byref(device), dev_ordinal))

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        self.cuda_check(
            self.cuda.cuDeviceComputeCapability(
                ctypes.byref(cc_major), ctypes.byref(cc_minor), device
            )
        )
        return cc_major.value, cc_minor.value

    def get_sm_thread_count(self, dev_ordinal: int = 0) -> Tuple[int, int]:
        device = ctypes.c_int()
        self.cuda_check(self.cuda.cuDeviceGet(ctypes.byref(device), dev_ordinal))

        sm_count = ctypes.c_int()
        self.cuda_check(
            self.cuda.cuDeviceGetAttribute(
                ctypes.byref(sm_count), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
            )
        )

        threads_per_sm = ctypes.c_int()
        self.cuda.cuDeviceGetAttribute(
            ctypes.byref(threads_per_sm),
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            device,
        )

        return sm_count.value, threads_per_sm.value


class CudaProfiler:
    def __init__(self):
        self._cudart = ctypes.CDLL("libcudart.so")

    def start(self):
        ret = self._cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception(f"cudaProfilerStart() returned {ret}")

    def stop(self):
        ret = self._cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception(f"cudaProfilerStop() returned {ret}")


if __name__ == "__main__":
    querier = CudaInfoQuerier()
    print(querier.gpu_ids)
    print(querier.get_compute_capability())
    print(querier.get_sm_thread_count())