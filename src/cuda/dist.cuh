#include <cuda_device_runtime_api.h>

__global__ void distortion_gather_lt32(unsigned int levels, double* training_sequence, double* codebook,
    double* error_matrix, unsigned int* cells, double* intermediate) {
    unsigned int idx = threadIdx.x + WARP_SIZE*blockIdx.x;
    double target = training_sequence[idx];
    unsigned int i_nnc = cells[idx];
    double sum = 0;
    // Perform summation
    for(int j = 0; j < levels; j++) {
        sum += error_matrix[j + levels*i_nnc] * (target - codebook[j]) * (target - codebook[j]);
    }
    intermediate[idx] = sum;
}

__global__ void distortion_gather_ge32(unsigned int levels, double* training_sequence, double* codebook,
    double* error_matrix, unsigned int* cells, double* intermediate) {
    extern __shared__ double s_codebook[];
    unsigned int t = threadIdx.x;
    unsigned int idx = threadIdx.x + WARP_SIZE*blockIdx.x;
    double target = training_sequence[idx];
    unsigned int loads_per_thread = levels / WARP_SIZE;
    unsigned int i_nnc = cells[idx];
    double sum = 0;
    // load codebook into shared mem
    for(unsigned int k = 0; k < loads_per_thread; k++) {
        s_codebook[loads_per_thread*t + k] = codebook[loads_per_thread*t + k];
    }
    // Perform summation
    for(int j = 0; j < levels; j++) {
        sum += error_matrix[j + levels*i_nnc] * (target - s_codebook[j]) * (target - s_codebook[j]);
    }
    intermediate[idx] = sum;
}