#include <cuda_device_runtime_api.h>

/*
    Each block handles 1 codebook index
*/
__global__ void cc_ge32(unsigned int levels, double* codebook, double* error_matrix, double* cc_cell_sums, unsigned int* cc_cardinality) {
    unsigned int t = threadIdx.x;
    unsigned int sums_per_thread = levels / WARP_SIZE;
    unsigned int j = blockIdx.x;
    double numerator = 0;
    double denominator = 0;
    // Perform summation
    for(unsigned int i = t*sums_per_thread; i < (t+1)*sums_per_thread; i++) {
        numerator += error_matrix[j + levels*i] * cc_cell_sums[i];
        denominator += error_matrix[j + levels*i] * cc_cardinality[i];
    }
    // Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if(t == 0)
        codebook[j] = numerator / denominator;
}