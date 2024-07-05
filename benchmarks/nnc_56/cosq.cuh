#include <cuda_device_runtime_api.h>
#define TRAINING_SIZE 1048576
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define max_tm_size (64*64) // Reserve max amount for transition matrix possible.

__constant__ double c_q_points[64]; // 64 x 64 transition matrix.
__constant__ double tm[max_tm_size]; // 64 x 64 transition matrix.

/**
 * Nearest neighbour condition for levels >= 32 && >= blockDim.x.
 *
 * Each thread in the block calculates 1 or more summation(s) for the NNC condition
 * A warp-level min reduction is performed across all warps, which is
 * then followed up by a final min reduction by a single warp.
 */
__global__ void nnc1(unsigned int levels, double* training_sequence, unsigned int* cells) {
    unsigned int reduction_size = blockDim.x / warpSize; // Number of elements to be reduced by final warp
    extern __shared__ char smem[];
    double* s_sums = (double*) smem;
    unsigned int* s_idx = (unsigned int*) (smem + reduction_size * sizeof(double));
    unsigned int t = threadIdx.x;
    double target = training_sequence[blockIdx.x];
    double sum = 0;
    unsigned int min_index;
    double min_sum = __DBL_MAX__;
    // Calculate NNC condition
    for(unsigned int k = 0; k < levels / blockDim.x; k++) {
        int l = t + k * blockDim.x;
        for(unsigned int i = 0; i < levels; i++) {
            // Transposed access: p(j|i) = mat[i + n*j] (coalesced access!)
            sum += tm[l + i * levels] * (target - c_q_points[i]) * (target - c_q_points[i]);
        }
        if(min_sum > sum) {
            min_sum = sum;
            min_index = l;
        }
        sum = 0;
    }
    __syncthreads();
    // min reduction on block.
    // Each warp in block does a reduction
    unsigned int shfl_min_index;
    double shfl_min_sum;
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
        shfl_min_sum = __shfl_down_sync(FULL_MASK, min_sum, offset);
        if(min_sum > shfl_min_sum) {
            min_index = shfl_min_index;
            min_sum = shfl_min_sum;
        }
    }
    if(t % warpSize == 0) {
        s_sums[t / warpSize] = min_sum;
        s_idx[t / warpSize] = min_index;
    }
    __syncthreads();
    min_sum = (t < reduction_size) ? s_sums[t] : 0;
    min_index = (t < reduction_size) ? s_idx[t] : 0;
    // Now reduce final warp
    if(t / warpSize == 0) {
        #pragma unroll
        for(int offset = reduction_size / 2; offset > 0; offset /= 2) {
            shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
            shfl_min_sum = __shfl_down_sync(FULL_MASK, min_sum, offset);
            if(min_sum > shfl_min_sum) {
                min_index = shfl_min_index;
                min_sum = shfl_min_sum;
            }
        }
    }
    if(t == 0) {
        cells[blockIdx.x] = min_index;
    }
}

/**
 * Nearest neighbour condition for levels >= 32 && < blockDim.x.
 *
 * Each thread in the block calculates 1 summation for the NNC condition
 * A warp-level min reduction is performed across all warps, which is
 * then followed up by a final min reduction by a single warp.
 *
 * Since there are <levels> distinct summations, the warp-level reductions have to be organized
 */
__global__ void nnc2(unsigned int levels, double* training_sequence, unsigned int* cells) {
    extern __shared__ char smem[];
    double* s_sums = (double*) smem;
    unsigned int* s_idx = (unsigned int*) (smem + (blockDim.x / warpSize) * sizeof(double));
    unsigned int t = threadIdx.x;
    unsigned int r = blockDim.x / levels;
    unsigned int e = t / levels;
    double target = training_sequence[r*blockIdx.x + e];
    double min_sum = 0;
    unsigned int min_index;
    unsigned int l = t % levels;
    for(unsigned int i = 0; i < levels; i++) {
        // Transposed access: p(j|i) = mat[i + n*j] (coalesced access!)
        min_sum += tm[l + i * levels] * (target - c_q_points[i]) * (target - c_q_points[i]);
    }
    min_index = l;
    __syncthreads();
    // Reduce
    unsigned int shfl_min_index;
    double shfl_min_sum;
    #pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
        shfl_min_sum = __shfl_down_sync(FULL_MASK, min_sum, offset);
        if(min_sum > shfl_min_sum) {
            min_index = shfl_min_index;
            min_sum = shfl_min_sum;
        }
    }
    if(t % warpSize == 0) {
        s_sums[t / warpSize] = min_sum;
        s_idx[t / warpSize] = min_index;
    }
    __syncthreads();
    unsigned int reduction_size = levels / warpSize;
    min_sum = l < reduction_size ? s_sums[l + reduction_size*e] : 0;
    min_index = l < reduction_size ? s_idx[l + reduction_size*e] : 0;
    // Now reduce final warp
    if(l < reduction_size) {
        #pragma unroll
        for(int offset = reduction_size / 2; offset > 0; offset /= 2) {
            shfl_min_index = __shfl_down_sync(FULL_MASK, min_index, offset);
            shfl_min_sum = __shfl_down_sync(FULL_MASK, min_sum, offset);
            if(min_sum > shfl_min_sum) {
                min_index = shfl_min_index;
                min_sum = shfl_min_sum;
            }
        }
    }
    if(l == 0) {
        cells[r*blockIdx.x + e] = min_index;
    }
}

/**
 * 1 block per codebook element
 * blockDim.x >= 32 permitted.
 * The more threads, the less 'scanning' work each thread has to do.
 */
__global__ void cc_p1(double* training_sequence, unsigned int* cells, double* cc_sums, unsigned int* cc_cardinality) {
    unsigned int f_reduce_count = blockDim.x / warpSize; // Number of elements to be reduced by final warp
    extern __shared__ char smem[];
    double* s_sums = (double*) smem;
    unsigned int* s_card = (unsigned int*) (smem + f_reduce_count * sizeof(double));
    unsigned int target = blockIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int scans = TRAINING_SIZE / blockDim.x;
    unsigned int cardinality = 0;
    double cell_sum = 0;
    for(int k = 0; k < scans; k++) {
        if(cells[t + k * blockDim.x] == target) {
            cardinality++;
            cell_sum += training_sequence[t + k * blockDim.x];
        }
    }
    __syncthreads();
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