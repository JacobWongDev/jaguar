#include <cuda_device_runtime_api.h>
#define WARP_SIZE 32

/*
    Each block handles 32 sums.
*/
template <unsigned int levels>
__global__ void distortion_gather(double* training_sequence, double* codebook, double* error_matrix, unsigned int* cells, double* intermediate) {
    __shared__ double s_codebook[levels];
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

/*******************************************************
 * NVIDIA
 ******************************************************/

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

__device__ __forceinline__ double warpReduceSum(unsigned int mask, double mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const double *__restrict__ g_idata, double *__restrict__ g_odata, unsigned int n) {
  double *sdata = SharedMemory<double>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  double mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent = (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}