#include <cuda_device_runtime_api.h>
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

/*
    References
    - Kahan Summation: https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/chapter1.pdf
*/

/**
 * CUDA Kernel for the Nearest Neighbour Condition (nnc), for codebooks with greater than or equal to 32 elements.
 * - loads codebook into shared mem from global memory. Each thread loads <num_sums> elements
 * - Each block is assigned a training element. For each element <levels> sums must be computed,
 *   each involving a sum of <levels> elements.
 * - Each thread computes <num_sums> of these <levels> number of sums.
 * - Then once the sum is computed, the minimum of all the sums is found using a reduction.
 * - Since this kernel is meant to be run on ONE WARP specifically, no __syncthreads() is required.
*/
template <unsigned int levels>
__global__ void nnc_e32(float* training_sequence, float* codebook, float* error_matrix, unsigned int* cells, float* cc_sums, unsigned int* cc_cardinal) {
    __shared__ float s_codebook[levels];
    __shared__ float s_sums[levels];
    __shared__ unsigned int s_min_indices[WARP_SIZE];
    unsigned int t = threadIdx.x;
    float target = training_sequence[blockIdx.x];
    float sum = 0;
    float c = 0; // compensation
    float next, next_sum;
    unsigned int num_sums = levels / WARP_SIZE;
    // load codebook into shared mem
    for(unsigned int k = 0; k < num_sums; k++) {
        s_codebook[num_sums*t + k] = codebook[num_sums*t + k];
    }
    for(unsigned int k = 0; k < num_sums; k++) {
        // Kahan Summation
        for(unsigned int i = 0; i < levels; i++) {
            next = error_matrix[i + levels*(num_sums*t + k)] * (target - s_codebook[i]) * (target - s_codebook[i]) + c;
            next_sum = sum + next;
            c = next - (next_sum - sum);
            sum = next_sum;
        }
        s_sums[num_sums*t + k] = sum;
        sum = 0;
        c = 0;
    }
    // Now, find minimum array INDEX of all the sums.
    unsigned int min_index = t;
    for(unsigned int k = 1; k < num_sums; k++) {
        if(s_sums[min_index] > s_sums[WARP_SIZE*k + t]) {
            min_index = WARP_SIZE*k + t;
        }
    }
    s_min_indices[t] = min_index;
    if(t < 16) {
        if(s_sums[s_min_indices[t]] > s_sums[s_min_indices[t + 16]]) {
            s_min_indices[t] = s_min_indices[t + 16];
        }
    }
    if(t < 8) {
        if(s_sums[s_min_indices[t]] > s_sums[s_min_indices[t + 8]]) {
            s_min_indices[t] = s_min_indices[t + 8];
        }
    }
    if(t < 4) {
        if(s_sums[s_min_indices[t]] > s_sums[s_min_indices[t + 4]]) {
            s_min_indices[t] = s_min_indices[t + 4];
        }
    }
    if(t < 2) {
        if(s_sums[s_min_indices[t]] > s_sums[s_min_indices[t + 2]]) {
            s_min_indices[t] = s_min_indices[t + 2];
        }
    }
    if(t == 0) {
        if(s_sums[s_min_indices[0]] > s_sums[s_min_indices[1]]) {
            s_min_indices[0] = s_min_indices[1];
        }
        cells[blockIdx.x] = s_min_indices[0];
        atomicAdd(cc_sums + s_min_indices[0], target);
        atomicAdd(cc_cardinal + s_min_indices[0], 1);
    }
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
template <unsigned int levels>
__global__ void nnc_e32_v2(float* training_sequence, float* codebook, float* error_matrix, unsigned int* cells, float* cc_sums, unsigned int* cc_cardinal) {
    __shared__ float s_codebook[levels];
    __shared__ float s_sums[levels];
    unsigned int t = threadIdx.x;
    float target = training_sequence[blockIdx.x];
    float sum = 0;
    unsigned int num_sums = levels / WARP_SIZE;
    unsigned int min_index = t;
    unsigned int shfl_min_index;
    float c = 0; // compensation
    float next, next_sum;
    // load codebook into shared mem
    for(unsigned int k = 0; k < num_sums; k++) {
        s_codebook[num_sums*t + k] = codebook[num_sums*t + k];
    }
    for(unsigned int k = 0; k < num_sums; k++) {
        // Kahan Summation
        for(unsigned int i = 0; i < levels; i++) {
            next = error_matrix[i + levels*(num_sums*t + k)] * (target - s_codebook[i]) * (target - s_codebook[i]) + c;
            next_sum = sum + next;
            c = next - (next_sum - sum);
            sum = next_sum;
        }
        s_sums[num_sums*t + k] = sum;
        sum = 0;
        c = 0;
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
        atomicAdd(cc_sums + min_index, target);
        atomicAdd(cc_cardinal + min_index, 1);
    }
}