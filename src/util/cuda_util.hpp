#include <cuda_runtime_api.h>

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


#define checkCudaErrors(error) check(error, __FILE__, __LINE__);
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
void check(cudaError_t error, const char* file, int line);

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
