#include <cuda_device_runtime_api.h>
#define FULL_MASK 0xffffffff

/**
 * @brief Nearest neighbour condition bit rates greater or equal to 5
 *
 * Each thread in the block calculates 1 summation for the NNC condition
 * A warp-level min reduction is performed across all warps, which is
 * then followed up by a final min reduction by a single warp.
 *
 * Since there are <levels> distinct summations, the warp-level reductions have to be organized.
 *
 * Since the maximum bit rate is 10, the largest possible reduction
 * size will be 2^10, and hence 2 iterations of warp reductions can
 * reduce the sum (assuming each warp reduces 32 elements).
 *
 * Kernel Requirements
 * levels >= 32 && levels < blockDim.x. blockDim.x is a power of 2.
 */
__global__ void nnc_ge5(unsigned int levels, double* training_sequence, const double* ctm, double* q_points, unsigned int* cells) {
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
        min_sum += ctm[l + i * levels] * (target - q_points[i]) * (target - q_points[i]);
    }
    min_index = l;
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
 * @brief Nearest neighbour condition bit rates less than 5.
 *
 * Since <levels> is less than 32, each warp has handles <blockDim.x / levels> codebook elements.
 * A reduction is performed on each warp, where within the warp <blockDim.x / levels> sized reductions
 * are performed.
 *
 * Since the maximum bit rate is 4, the largest possible reduction
 * size will be 16, and hence multiple small reductions within the warp can
 * reduce the sum (assuming each warp reduces 32 elements).
 *
 * Kernel Requirements
 * levels <= 16 && blockDim.x >= 32. blockDim.x is a power of 2.
 */
__global__ void nnc_lt5(unsigned int levels, double* training_sequence, double* q_points, double* ctm, unsigned int* cells) {
    unsigned int t = threadIdx.x;
    unsigned int l = t % levels;
    unsigned int r = blockDim.x / levels;
    unsigned int e = t / levels;
    double target = training_sequence[r*blockIdx.x + e];
    double min_sum = 0;
    unsigned int min_index;
    for(unsigned int i = 0; i < levels; i++) {
        // Transposed access: p(j|i) = mat[i + n*j] (coalesced access!)
        min_sum += ctm[l + i * levels] * (target - q_points[i]) * (target - q_points[i]);
    }
    min_index = l;
    // min reduction
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
        cells[r*blockIdx.x + e] = min_index;
    }
}