#include <cuda_device_runtime_api.h>
#define TRAINING_SIZE 1048576
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void nnc_br_sm(unsigned int levels, double* training_sequence, double* codebook,
        double* error_matrix, unsigned int* cells, double* cc_sums, unsigned int* cc_cardinal) {
    unsigned int t = threadIdx.x;
    unsigned int target_idx = t / levels + blockIdx.x * (WARP_SIZE / levels);
    double target = training_sequence[target_idx];
    unsigned int l = t % levels;
    double min_sum = 0;
    #pragma unroll
    for(unsigned int j = 0; j < levels; j++) {
        min_sum += error_matrix[j + levels*l] * (target - codebook[j]) * (target - codebook[j]);
    }
    // Minimum reduction
    unsigned int min_index = l;
    unsigned int shfl_min_index;
    double shfl_min_sum;

    #pragma unroll
    for(int offset = levels / 2; offset > 0; offset /= 2) {
        shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
        shfl_min_sum = __shfl_down_sync(FULL_MASK, min_sum, offset);
        if(min_sum > shfl_min_sum) {
            min_index = shfl_min_index;
            min_sum = shfl_min_sum;
        }
    }

    if(l == 0) {
        cells[target_idx] = min_index;
        atomicAddDouble(cc_sums + min_index, target);
        atomicAdd(cc_cardinal + min_index, 1);
    }
}