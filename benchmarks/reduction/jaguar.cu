#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "reduction.cuh"

#define TRAINING_SIZE (1 << 20)
#define MIN(a, b) ((a < b) ? a : b)
#define checkCudaErrors(error) check(error, __FILE__, __LINE__);

void check(cudaError_t error, const char* file, int line) {
  if(cudaSuccess != error) {
      printf("CUDA error in %s: line %d code=%d(%s): %s\n", file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
  }
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel For the kernels >= 3, we set threads / block to the minimum of
// maxThreads and n/2. For kernels < 3, we set to the minimum of maxThreads and
// n.  For kernel 6, we observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
  blocks = (n + (threads * 2 - 1)) / (threads * 2);

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }
  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }
  blocks = MIN(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
float reduceCPU(float *data, int size) {
  float sum = data[0];
  float c = 0.0f;

  for (int i = 1; i < size; i++) {
    float y = data[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

bool isPow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduce(int size, int threads, int blocks, float *device_seq, float* device_res) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
  // For reduce7 kernel we require only blockSize/warpSize
  // number of elements in shared memory
  smemSize = ((threads / 32) + 1) * sizeof(float);
  if(isPow2(size)) {
    switch (threads) {
      case 1024:
        reduce7<1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;
      case 512:
        reduce7<512, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 256:
        reduce7<256, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 128:
        reduce7<128, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 64:
        reduce7<64, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 32:
        reduce7<32, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 16:
        reduce7<16, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 8:
        reduce7<8, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 4:
        reduce7<4, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 2:
        reduce7<2, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 1:
        reduce7<1, true>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;
    }
  } else {
    switch (threads) {
      case 1024:
        reduce7<1024, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;
      case 512:
        reduce7<512, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 256:
        reduce7<256, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 128:
        reduce7<128, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 64:
        reduce7<64, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 32:
        reduce7<32, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 16:
        reduce7<16, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 8:
        reduce7<8, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 4:
        reduce7<4, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 2:
        reduce7<2, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;

      case 1:
        reduce7<1, false>
            <<<dimGrid, dimBlock, smemSize>>>(device_seq, device_res, size);
        break;
    }
  }
}

int main(int argc, char** argv) {
  float* sequence = (float*) malloc(sizeof(float) * TRAINING_SIZE);
  unsigned int maxThreads = 256;  // number of threads per block
  unsigned int maxBlocks = 64;
  int cpuFinalThreshold = 1;
  bool needReadBack = true;
  int threads, blocks;
  getNumBlocksAndThreads(TRAINING_SIZE, maxBlocks, maxThreads, blocks, threads);
  // generate random data
  for (int i = 0; i < TRAINING_SIZE; i++) {
    // Keep the numbers small so we don't get truncation error in the sum
    sequence[i] = (rand() & 0xFF) / (float) RAND_MAX;
  }
  float* result = (float*) malloc(sizeof(float) * blocks);
  float* device_seq;
  float* device_res;
  checkCudaErrors(cudaMalloc((void **)&device_seq, sizeof(float) * TRAINING_SIZE));
  checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * blocks));
  checkCudaErrors(cudaMemcpy(device_seq, sequence, sizeof(float) * TRAINING_SIZE, cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(device_res, result, sizeof(float) * blocks, cudaMemcpyHostToDevice));

  // Perform GPU reduction
  float* device_intermediate;
  float gpu_res=0;
  checkCudaErrors(cudaMalloc((void **)&device_intermediate, sizeof(float) * blocks));

  reduce(TRAINING_SIZE, threads, blocks, device_seq, device_res);

  int s = blocks;
  while(s > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
    checkCudaErrors(cudaMemcpy(device_intermediate, device_res, s * sizeof(float), cudaMemcpyDeviceToDevice));
    reduce(s, threads, blocks, device_intermediate, device_res);
    s = (s + (threads * 2 - 1)) / (threads * 2);
  }

  if (s > 1) {
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(result, device_res, s * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < s; i++) {
      gpu_res += result[i];
    }
    needReadBack = false;
  }

  if (needReadBack) {
    // copy final sum from device to host
    checkCudaErrors(cudaMemcpy(&gpu_res, device_res, sizeof(float), cudaMemcpyDeviceToHost));
  }

  // compare results
  unsigned int precision = 8;
  double threshold = 1e-8 * TRAINING_SIZE;
  float cpu_res = reduceCPU(sequence, TRAINING_SIZE);
  double diff = fabs((double)gpu_res - (double)cpu_res);

  if(diff < threshold) {
    printf("Test passed!\n");
  } else {
    printf("Test failed!\n");
  }

  printf("\nGPU result = %.*f\n", precision, (double)gpu_res);
  printf("CPU result = %.*f\n\n", precision, (double)cpu_res);

  checkCudaErrors(cudaFree(device_seq));
  checkCudaErrors(cudaFree(device_res));
  checkCudaErrors(cudaFree(device_intermediate));
  free(sequence);
  free(result);
}