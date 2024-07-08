#include <cuda_device_runtime_api.h>

/**
 * 1 block per q_points element
 * blockDim.x >= 32 permitted.
 * The more threads, the less 'scanning' work each thread has to do.
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

/*
    Each block computes 1 q_points element
    and levels >= blockDim.x.
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

/*
    Each block computes 1 q_points element
    and levels >= blockDim.x.
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