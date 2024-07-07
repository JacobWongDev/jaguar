#include <cuda_device_runtime_api.h>
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#include <stdio.h>

/*
    Each block computes 1 codebook element
    and levels >= blockDim.x.
*/
__global__ void cc_gt5(unsigned int levels, double* codebook, double* error_matrix,
        double* cc_cell_sums, unsigned int* cc_cardinality) {
    extern __shared__ char smem[];
    double* s_numerator = (double*) smem;
    unsigned int s = blockDim.x / warpSize; // reduction size
    double* s_denominator = (double*) (smem + s * sizeof(double));
    unsigned int t = threadIdx.x;
    unsigned int r = levels / blockDim.x;
    double numerator = 0;
    double denominator = 0;
    for(int k = 0; k < r; k++) {
        int l = t + blockDim.x * k;
        numerator += error_matrix[l + levels * blockIdx.x] * cc_cell_sums[l];
        denominator += error_matrix[l + levels * blockIdx.x] * cc_cardinality[l];
    }
    // reduce block
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if(t % warpSize == 0) {
        s_numerator[t / warpSize] = numerator;
        s_denominator[t / warpSize] = denominator;
    }
    __syncthreads();
    numerator = (t < s) ? s_numerator[t] : 0;
    denominator = (t < s) ? s_denominator[t] : 0;
    // Now reduce final warp
    if(t / warpSize == 0) {
        #pragma unroll
        for(int offset = s / 2; offset > 0; offset /= 2) {
            numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
            denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
        }
    }
    if(t == 0) {
        codebook[blockIdx.x] = numerator / denominator;
    }
}

/*
    Each block computes 1 codebook element
    and levels >= blockDim.x.
*/
__global__ void cc_le5(unsigned int levels, double* codebook, double* error_matrix,
        double* cc_cell_sums, unsigned int* cc_cardinality) {
    unsigned int t = threadIdx.x;
    unsigned int r = levels / blockDim.x;
    double numerator = 0;
    double denominator = 0;
    for(int k = 0; k < r; k++) {
        int l = t + r * k;
        numerator += error_matrix[l + levels * blockIdx.x] * cc_cell_sums[l];
        denominator += error_matrix[l + levels * blockIdx.x] * cc_cardinality[l];
    }
    // Now reduce warp
    #pragma unroll
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if(t == 0) {
        codebook[blockIdx.x] = numerator / denominator;
    }
}