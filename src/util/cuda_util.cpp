/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_util.h"
#include "spdlog/spdlog.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

sSMtoCores nGpuArchCoresPerSM[] = {
    {0x30, 192},
    {0x32, 192},
    {0x35, 192},
    {0x37, 192},
    {0x50, 128},
    {0x52, 128},
    {0x53, 128},
    {0x60,  64},
    {0x61, 128},
    {0x62, 128},
    {0x70,  64},
    {0x72,  64},
    {0x75,  64},
    {0x80,  64},
    {0x86, 128},
    {0x87, 128},
    {0x89, 128},
    {0x90, 128},
    {-1, -1}
};

sSMtoArchName nGpuArchNameSM[] = {
    {0x30, "Kepler"},
    {0x32, "Kepler"},
    {0x35, "Kepler"},
    {0x37, "Kepler"},
    {0x50, "Maxwell"},
    {0x52, "Maxwell"},
    {0x53, "Maxwell"},
    {0x60, "Pascal"},
    {0x61, "Pascal"},
    {0x62, "Pascal"},
    {0x70, "Volta"},
    {0x72, "Xavier"},
    {0x75, "Turing"},
    {0x80, "Ampere"},
    {0x86, "Ampere"},
    {0x87, "Ampere"},
    {0x89, "Ada"},
    {0x90, "Hopper"},
    {-1, "Graphics Device"}
};

void check(cudaError_t error, const char* file, int line) {
    if(cudaSuccess != error) {
        spdlog::error("CUDA error in {:s}: line {:d} code={:d}({:s}): {:s}\n",
                file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
    }
}

int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    spdlog::error("MapSMtoCores for SM {:d}.{:d} is undefined.", major, minor);
    spdlog::error("Default to use {:d} Cores/SM", nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

const char* _ConvertSMVer2ArchName(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the GPU Arch name)
    int index = 0;
    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one
    // to run properly
    spdlog::error("MapSMtoArchName for SM {:d}.{:d} is undefined.", major, minor);
    spdlog::error("Default to use {:s}", nGpuArchNameSM[index - 1].name);
    return nGpuArchNameSM[index - 1].name;
}

/**
 * @brief Finds a CUDA-enabled device for computations.
 *
 */
int gpuGetMaxGflopsDeviceId() {
    int current_device = 0;
    int sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;
    uint64_t max_compute_perf = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        spdlog::error("gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA!\n");
    }
    // Find the best CUDA capable GPU device
    current_device = 0;
    while (current_device < device_count) {
        int computeMode = -1, major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited) {
            if (major == 9999 && minor == 9999) {
                sm_per_multiproc = 1;
            } else {
                sm_per_multiproc = _ConvertSMVer2Cores(major,  minor);
            }
            int multiProcessorCount = 0;
            int clockRate = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
            cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess) {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if(result == cudaErrorInvalidValue) {
                    clockRate = 1;
                } else {
                    checkCudaErrors(result);
                    return -1;
                }
            }
            uint64_t compute_perf = (uint64_t) multiProcessorCount * sm_per_multiproc * clockRate;
            if (compute_perf > max_compute_perf) {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        } else {
            devices_prohibited++;
        }
        ++current_device;
    }
    if (devices_prohibited == device_count) {
        spdlog::error("gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited!\n");
        return -1;
    }
    return max_perf_device;
}

bool cuda_init() {
    struct cudaDeviceProp properties;
    int device_id = gpuGetMaxGflopsDeviceId();
    if (device_id == -1)
        return false;
    checkCudaErrors(cudaSetDevice(device_id));
    int major = 0;
    int minor = 0;
    // size_t device_heap_size = (size_t) 2 << 31;
    // checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, device_heap_size));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    spdlog::info("Found GPU Device {:d}!", device_id);
    spdlog::info("Compute Architecture {:s}", _ConvertSMVer2ArchName(major, minor));
    spdlog::info("Compute Capability {:d}.{:d}", major, minor);
    checkCudaErrors(cudaGetDeviceProperties(&properties, device_id));
    spdlog::info("Device Properties:");
    spdlog::info("Device name: {:s}", properties.name);
    spdlog::info("Streaming Multiprocessors (SMs): {:d}", properties.multiProcessorCount);
    spdlog::info("Warp size: {:d}", properties.warpSize);
    spdlog::info("Maximum threads per block: {:d}", properties.maxThreadsPerBlock);
    spdlog::info("Total Global memory: {:f}GB", properties.totalGlobalMem * 1e-9);
    spdlog::info("Shared Memory per block: {:f}KB", properties.sharedMemPerBlock * 1e-3);
    spdlog::info("Registers per block: {:d}", properties.regsPerBlock);
    return true;
}