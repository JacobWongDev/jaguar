#include "ext.h"

#define MIN(a, b) ((a < b) ? a : b)

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

bool isPow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
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

  if ((double)threads * blocks >
      (double)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    spdlog::error("n is too large, please choose a smaller number!");
  }
  if (blocks > prop.maxGridSize[0]) {
    spdlog::error("Grid size <{:d}> exceeds the device capability <{:d}>, set block size as {:d} (original {:d})",
        blocks, prop.maxGridSize[0], threads * 2, threads);
    blocks /= 2;
    threads *= 2;
  }
  blocks = MIN(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduce(int size, int threads, int blocks, double *device_seq, double* device_res) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
  // For reduce7 kernel we require only blockSize/warpSize
  // number of elements in shared memory
  smemSize = ((threads / 32) + 1) * sizeof(double);
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

double distortion_reduce(unsigned int training_size, double* device_reduce_sums) {
  unsigned int maxThreads = 256;  // number of threads per block
  unsigned int maxBlocks = 64;
  int cpuFinalThreshold = 1;
  bool needReadBack = true;
  int threads, blocks;
  getNumBlocksAndThreads(training_size, maxBlocks, maxThreads, blocks, threads);
  double* device_res;
  double* result = (double*) malloc(sizeof(double) * blocks);
  checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(double) * blocks));
  // checkCudaErrors(cudaMemcpy(device_res, result, sizeof(double) * blocks, cudaMemcpyHostToDevice));

  // Perform GPU reduction
  double* device_intermediate;
  double gpu_res=0;
  checkCudaErrors(cudaMalloc((void **)&device_intermediate, sizeof(double) * blocks));

  reduce(training_size, threads, blocks, device_reduce_sums, device_res);

  int s = blocks;
  while(s > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
    checkCudaErrors(cudaMemcpy(device_intermediate, device_res, s * sizeof(double), cudaMemcpyDeviceToDevice));
    reduce(s, threads, blocks, device_intermediate, device_res);
    s = (s + (threads * 2 - 1)) / (threads * 2);
  }

  if (s > 1) {
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(result, device_res, s * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < s; i++) {
      gpu_res += result[i];
    }
    needReadBack = false;
  }

  if (needReadBack) {
    // copy final sum from device to host
    checkCudaErrors(cudaMemcpy(&gpu_res, device_res, sizeof(double), cudaMemcpyDeviceToHost));
  }
  free(result);
  checkCudaErrors(cudaFree(device_res));
  checkCudaErrors(cudaFree(device_intermediate));
  return gpu_res / training_size;
}