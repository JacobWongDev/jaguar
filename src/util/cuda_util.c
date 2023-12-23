#include <cuda_runtime_api.h>
#include "cuda_util.h"
#include "logger.h"
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

void checkCudaErrors(cudaError_t error) {
    if(cudaSuccess != error) {
        char buffer[50];
        snprintf(buffer, 50, "CUDA error: code=%d(%s)\n", (unsigned int) error, cudaGetErrorName(error));
        logger_send(buffer, ERROR);
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
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
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
    printf(
        "MapSMtoArchName for SM %d.%d is undefined."
        "  Default to use %s\n",
        major, minor, nGpuArchNameSM[index - 1].name);
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
        logger_send("gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA!\n", ERROR);
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
        logger_send("gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited!\n", ERROR);
        return -1;
    }
    return max_perf_device;
}

bool cuda_init() {
    int device_id = gpuGetMaxGflopsDeviceId();
    char success_message[100];
    if (device_id == -1)
        return false;
    checkCudaErrors(cudaSetDevice(device_id));
    int major = 0;
    int minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    snprintf(success_message, 100, "Found GPU Device %d! \"%s\" with compute capability %d.%d\n\n", device_id, _ConvertSMVer2ArchName(major, minor), major, minor);
    logger_send(success_message, INFO);
    return true;
}