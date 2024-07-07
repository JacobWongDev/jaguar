#include <cuda_device_runtime_api.h>
#define WARP_SIZE 32

/*
  Each block handles <blockDim.x> number of sums.
*/
__global__ void distortion_gather(unsigned int levels, double* training_sequence, double* codebook,
  double* error_matrix, unsigned int* cells, double* intermediate) {
  extern __shared__ double s_codebook[];
  unsigned int t = threadIdx.x;
  unsigned int t_ = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int r = levels / blockDim.x;
  double target = training_sequence[t_];
  unsigned int i_nnc = cells[t_];
  double sum = 0;
  if(r == 0) {
    if(t < levels) {
      s_codebook[t] = codebook[t];
    }
  } else {
    for(int i = 0; i < r; i++) {
      s_codebook[t + blockDim.x * i] = codebook[t + blockDim.x * i];
    }
  }
  __syncthreads();
  //Perform summation
  for(int j = 0; j < levels; j++) {
    sum += error_matrix[i_nnc + levels*j] * (target - s_codebook[j]) * (target - s_codebook[j]);
  }
  // for(int j = 0; j < levels; j++) {
  //   sum += error_matrix[i_nnc + levels*j] * (target - codebook[j]) * (target - codebook[j]);
  // }
  intermediate[t_] = sum;
}

/*
  Each block handles <blockDim.x> number of sums.
*/
// __global__ void distortion_gather(unsigned int levels, double* training_sequence, double* codebook,
//   double* error_matrix, unsigned int* cells, double* intermediate) {
//   unsigned int t = threadIdx.x + blockIdx.x * blockDim.x;
//   double target = training_sequence[t];
//   unsigned int i_nnc = cells[t];
//   double sum = 0;
//   //Perform summation
//   for(int j = 0; j < levels; j++) {
//     sum += error_matrix[i_nnc + levels*j] * (target - codebook[j]) * (target - codebook[j]);
//   }
//   intermediate[t] = sum;
// }

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