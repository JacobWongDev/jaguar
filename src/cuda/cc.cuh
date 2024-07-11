#include <cuda_device_runtime_api.h>

/**
 * @brief 1st half of Centroid Condition.
 * 2nd half is cc_* kernel.
 *
 * This kernel computes the cardinality of each quantization cell, and the sum of each cell.
 * The more threads, the less 'scanning' work each thread has to do.
 * Each thread 'scans' part of the cells array to look for its assigned codebook index.
 *
 * Kernel Requirements:
 * blockDim.x >= 32, and blockDim.x is a power of 2.
 */
__global__ void cc_gather(double* training_sequence, unsigned int training_size,
        unsigned int* cells, double* cc_sums, unsigned int* cc_cardinality) {
    unsigned int f_reduce_count = blockDim.x / warpSize; // Number of elements to be reduced by final warp
    extern __shared__ char smem[];
    double* s_sums = (double*) smem;
    unsigned int* s_card = (unsigned int*) (smem + f_reduce_count * sizeof(double));
    unsigned int target = blockIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int scans = training_size / blockDim.x;
    unsigned int cardinality = 0;
    double cell_sum = 0;
    for(int k = 0; k < scans; k++) {
        if(cells[t + k * blockDim.x] == target) {
            cardinality++;
            cell_sum += training_sequence[t + k * blockDim.x];
        }
    }
    // reduce block
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        cell_sum += __shfl_down_sync(FULL_MASK, cell_sum, offset);
        cardinality += __shfl_down_sync(FULL_MASK, cardinality, offset);
    }
    if(t % warpSize == 0) {
        s_sums[t / warpSize] = cell_sum;
        s_card[t / warpSize] = cardinality;
    }
    __syncthreads();
    cell_sum = (t < f_reduce_count) ? s_sums[t] : 0;
    cardinality = (t < f_reduce_count) ? s_card[t] : 0;
    // Now reduce final warp
    if(t / warpSize == 0) {
        #pragma unroll
        for(int offset = f_reduce_count / 2; offset > 0; offset /= 2) {
            cell_sum += __shfl_down_sync(FULL_MASK, cell_sum, offset);
            cardinality += __shfl_down_sync(FULL_MASK, cardinality, offset);
        }
    }
    if(t == 0) {
        cc_sums[blockIdx.x] = cell_sum;
        cc_cardinality[blockIdx.x] = cardinality;
    }
}

/**
 * @brief 2nd half of Centroid Condition, bit rate greater or equal to 5.
 *
 * This kernel computes the new quantization points (q_points).
 * Each block computes 1 quantization point using the cardinality and
 * cell sums from cc_gather.
 *
 * Since the maximum bit rate is 10, the largest possible reduction
 * size will be 2^10, and hence 2 iterations of warp reductions can
 * reduce the sum (assuming each warp reduces 32 elements).
 *
 * Kernel Requirements:
 * levels >= blockDim.x, and blockDim.x is a power of 2.
 */
__global__ void cc_ge5(unsigned int levels, double* q_points, double* ctm,
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
        numerator += ctm[l + levels * blockIdx.x] * cc_cell_sums[l];
        denominator += ctm[l + levels * blockIdx.x] * cc_cardinality[l];
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
        q_points[blockIdx.x] = numerator / denominator;
    }
}

/**
 * @brief 2nd half of Centroid Condition, bit rate less than 5.
 *
 * This kernel computes the new quantization points (q_points).
 * Each block computes 1 quantization point using the cardinality and
 * cell sums from cc_gather.
 *
 * Since the bit rate < 5, the largest possible reduction
 * size will be 32, and hence 1 warp reduction can
 * reduce the sum (assuming each warp reduces 32 elements).
 *
 * Kernel Requirements:
 * levels >= blockDim.x, and blockDim.x is a power of 2.
 */
__global__ void cc_le5(unsigned int levels, double* q_points, double* ctm,
        double* cc_cell_sums, unsigned int* cc_cardinality) {
    unsigned int t = threadIdx.x;
    unsigned int r = levels / blockDim.x;
    double numerator = 0;
    double denominator = 0;
    for(int k = 0; k < r; k++) {
        int l = t + r * k;
        numerator += ctm[l + levels * blockIdx.x] * cc_cell_sums[l];
        denominator += ctm[l + levels * blockIdx.x] * cc_cardinality[l];
    }
    // Now reduce warp
    #pragma unroll
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if(t == 0) {
        q_points[blockIdx.x] = numerator / denominator;
    }
}