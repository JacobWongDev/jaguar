#include <cuda_device_runtime_api.h>
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

/**
 * CUDA Kernel for the Nearest Neighbour Condition (nnc), for codebooks with greater than or equal to 32 elements.
 * - loads codebook into shared mem from global memory. Each thread loads <num_sums> elements
 * - Each block is assigned a training element. For each element <levels> sums must be computed,
 *   each involving a sum of <levels> elements.
 * - Each thread computes <num_sums> of these <levels> number of sums.
 * - Then once the sum is computed, the minimum of all the sums is found using a reduction.
 * - Since this kernel is meant to be run on ONE WARP specifically, no __syncthreads() is required.
*/
__global__ void nnc(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix,
        unsigned int* cells, double* cc_sums, unsigned int* cc_cardinal) {
    extern __shared__ double smem[];
    double* s_codebook = smem;
    double* s_sums = smem + levels;
    unsigned int t = threadIdx.x;
    double target = training_sequence[blockIdx.x];
    double sum = 0;
    unsigned int num_sums = levels / WARP_SIZE;
    unsigned int min_index = t;
    unsigned int shfl_min_index;
    // load codebook into shared mem
    for(unsigned int k = 0; k < num_sums; k++) {
        s_codebook[num_sums*t + k] = codebook[num_sums*t + k];
    }
    for(unsigned int k = 0; k < num_sums; k++) {
        for(unsigned int i = 0; i < levels; i++) {
            sum += error_matrix[i + levels*(num_sums*t + k)] * (target - s_codebook[i]) * (target - s_codebook[i]);
        }
        s_sums[num_sums*t + k] = sum;
        sum = 0;
    }
    // Now, find minimum array INDEX of all the sums.
    for(unsigned int k = 1; k < num_sums; k++) {
        if(s_sums[min_index] > s_sums[WARP_SIZE*k + t]) {
            min_index = WARP_SIZE*k + t;
        }
    }
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2) {
        shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
        if(s_sums[min_index] > s_sums[shfl_min_index]) {
            min_index = shfl_min_index;
        }
    }
    if(t == 0) {
        cells[blockIdx.x] = min_index;
        atomicAddDouble(cc_sums + min_index, target);
        atomicAdd(cc_cardinal + min_index, 1);
    }
}