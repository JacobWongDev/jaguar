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

__global__ void nnc_lt32(unsigned int levels, double* training_sequence, double* codebook,
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

__global__ void nnc_ge32(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix,
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

__global__ void s_nnc_lt32(unsigned int levels, double* training_sequence, double* codebook,
        double* error_matrix, double* cc_sums, unsigned int* cc_cardinal) {
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
        atomicAddDouble(cc_sums + min_index, target);
        atomicAdd(cc_cardinal + min_index, 1);
    }
}

__global__ void s_nnc_ge32(unsigned int levels, double* training_sequence, double* codebook, double* error_matrix,
        double* cc_sums, unsigned int* cc_cardinal) {
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
        atomicAddDouble(cc_sums + min_index, target);
        atomicAdd(cc_cardinal + min_index, 1);
    }
}