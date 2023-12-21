#include <stdbool.h>
#include <cuda_runtime_api.h>

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
} sSMtoCores;

extern sSMtoCores nGpuArchCoresPerSM[];

typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
} sSMtoArchName;

extern sSMtoArchName nGpuArchNameSM[];

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
void checkCudaErrors(cudaError_t error);

int _ConvertSMVer2Cores(int major, int minor);

const char* _ConvertSMVer2ArchName(int major, int minor);

/**
 * @brief Finds CUDA Device ID with maximum GFlops/s.
 *
 */
int gpuGetMaxGflopsDeviceId();

/**
 * @brief Finds a CUDA-enabled device for computations.
 *
 */
bool cuda_init();

#endif