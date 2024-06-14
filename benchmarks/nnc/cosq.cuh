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

/**
 * CUDA Kernel for the Nearest Neighbour Condition (nnc), for codebooks with greater than or equal to 32 elements.
 * - loads codebook into shared mem from global memory. Each thread loads <num_sums> elements
 * - Each block is assigned a training element. For each element <levels> sums must be computed,
 *   each involving a sum of <levels> elements.
 * - Each thread computes <num_sums> of these <levels> number of sums.
 * - Then once the sum is computed, the minimum of all the sums is found using a reduction.
 * - Since this kernel is meant to be run on ONE WARP specifically, no __syncthreads() is required.
*/
__global__ void nnc_e32(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix, unsigned int* cells, double* cc_sums, unsigned int* cc_cardinal) {
    extern __shared__ double smem[];
    double* s_codebook = smem;
    double* s_sums = smem + levels;
    __shared__ unsigned int s_min_indices[WARP_SIZE];
    unsigned int t = threadIdx.x;
    double target = training_sequence[blockIdx.x];
    double sum = 0;
    unsigned int num_sums = levels / WARP_SIZE;
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
        atomicAddDouble(cc_sums + s_min_indices[0], target);
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
__global__ void nnc_e32_v2(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix, unsigned int* cells, double* cc_sums, unsigned int* cc_cardinal) {
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

__global__ void nnc_e32_v3(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix, unsigned int* cells) {
    extern __shared__ double smem[];
    double* s_codebook = smem;
    double* s_sums = smem + levels;
    unsigned int t = threadIdx.x;
    double target = training_sequence[blockIdx.x];
    double sum = 0;
    double c = 0;
    unsigned int num_sums = levels / WARP_SIZE;
    unsigned int min_index = t;
    unsigned int shfl_min_index;
    // load codebook into shared mem
    for(unsigned int k = 0; k < num_sums; k++) {
        s_codebook[num_sums*t + k] = codebook[num_sums*t + k];
    }
    // Kahan Summation
    for(unsigned int k = 0; k < num_sums; k++) {
        for(unsigned int i = 0; i < levels; i++) {
            double y = error_matrix[i + levels*(num_sums*t + k)] * (target - s_codebook[i]) * (target - s_codebook[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
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
    }
}

template <unsigned int levels>
__global__ void cc_p1(double* training_sequence, unsigned int* cells, double* cc_sums, unsigned int* cc_cardinality) {
    unsigned int target = blockIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int section = TRAINING_SIZE / WARP_SIZE;
    unsigned int cardinality = 0;
    double cell_sum = 0;
    double c = 0;
    // Kahan Summation
    for(int i = 0; i < section; i++) {
        if(cells[i + section*t] == target) {
            cardinality++;
            double y = training_sequence[i + section*t] - c;
            double t = cell_sum + y;
            c = (t - cell_sum) - y;
            cell_sum = t;
        }
    }
    // reduce
    for(int offset = 16; offset > 0; offset /= 2) {
        cardinality += __shfl_down_sync(FULL_MASK, cardinality, offset);
        cell_sum += __shfl_down_sync(FULL_MASK, cell_sum, offset);
    }
    if(t == 0) {
        cc_cardinality[target] = cardinality;
        cc_sums[target] = cell_sum;
    }
}