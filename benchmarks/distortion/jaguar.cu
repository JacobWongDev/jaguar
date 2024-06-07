#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "cosq.cuh"

#define TRAINING_SIZE (1 << 20)
#define RATE 8
#define POLYA_EPSILON 0.01
#define POLYA_DELTA 0
#define FLOAT_ERROR 1

void check(cudaError_t error, const char* file, int line) {
  if(cudaSuccess != error) {
      printf("CUDA error in %s: line %d code=%d(%s): %s\n", file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
  }
}

#define checkCudaErrors(error) check(error, __FILE__, __LINE__);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float distortion_cpu(unsigned int levels, float* training_sequence, float* error_matrix, float* codebook, unsigned int* cells) {
  float d = 0;
  float c = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    for(int j = 0; j < levels; j++) {
      float y = error_matrix[j + levels*cells[i]] * (training_sequence[i] - codebook[j]) * (training_sequence[i] - codebook[j]) - c;
      float t = d + y;
      c = (t - d) - y;
      d = t;
    }
  }
  return d / TRAINING_SIZE;
}


inline float polya_urn_error(int j, int i, int num_bits) {
  float temp;
  int x = j ^ i;
  int previous;
  if(x & 1 == 1) {
    temp = POLYA_EPSILON;
    previous = 1;
  } else {
    temp = 1 - POLYA_EPSILON;
    previous = 0;
  }
  x >>= 1;
  for(int i = 1; i < num_bits; i++) {
    if(x & 1 == 1) {
      temp *= (POLYA_EPSILON + previous * POLYA_DELTA) / (1 + POLYA_DELTA);
      previous = 1;
    } else {
      temp *= ((1 - POLYA_EPSILON) + (1 - previous)*POLYA_DELTA) / (1 + POLYA_DELTA);
      previous = 0;
    }
    x >>= 1;
  }
  return temp;
}

float* compute_error_matrix(unsigned int levels) {
  float* error_matrix = (float*) malloc(sizeof(float) * levels * levels);
  for(int i = 0; i < levels; i++) {
      for(int j = 0; j < levels; j++) {
          error_matrix[j + i * levels] = polya_urn_error(j, i, RATE);
      }
  }
  return error_matrix;
}

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
float* generate_normal_sequence() {
  float* normal_sequence = (float*) malloc(TRAINING_SIZE * sizeof(float));
  std::default_random_engine rng;
  rng.seed(31);
  std::normal_distribution<float> distribution(10, 1);
  for(int i = 0; i < TRAINING_SIZE; i++) {
      normal_sequence[i] = distribution(rng);
  }
  return normal_sequence;
}

float distortion_reduce(float* device_reduce_sums) {
  unsigned int maxThreads = 256;  // number of threads per block
  unsigned int maxBlocks = 64;
  int cpuFinalThreshold = 1;
  bool needReadBack = true;
  int threads, blocks;
  getNumBlocksAndThreads(TRAINING_SIZE, maxBlocks, maxThreads, blocks, threads);
  float* device_res;
  float* result = (float*) malloc(sizeof(float) * blocks);
  checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * blocks));
  // checkCudaErrors(cudaMemcpy(device_res, result, sizeof(float) * blocks, cudaMemcpyHostToDevice));

  // Perform GPU reduction
  float* device_intermediate;
  float gpu_res=0;
  checkCudaErrors(cudaMalloc((void **)&device_intermediate, sizeof(float) * blocks));

  reduce(TRAINING_SIZE, threads, blocks, device_reduce_sums, device_res);

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
  free(result);
  checkCudaErrors(cudaFree(device_res));
  checkCudaErrors(cudaFree(device_intermediate));
  return gpu_res / TRAINING_SIZE;
}

int main(int argc, char** argv) {
  const unsigned int levels = 1 << RATE;
  float* training_sequence = generate_normal_sequence();
  float* error_matrix = compute_error_matrix(levels);
  float* codebook = (float*) malloc(sizeof(float) * levels);
  unsigned int* cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);;
  // intialize codebook to first <levels> training samples
  std::default_random_engine rng;
  rng.seed(31);
  std::uniform_int_distribution<int> distribution(0, levels - 1);
  for(int i = 0; i < levels; i++) {
    codebook[i] = training_sequence[i];
  }
  for(int i = 0; i < TRAINING_SIZE; i++) {
    cells[i] = distribution(rng);
  }
  /*
    Sequential distortion
  */
  float d1 = 0;
  auto start = std::chrono::high_resolution_clock::now();
  d1 = distortion_cpu(levels, training_sequence, error_matrix, codebook, cells);
  auto end = std::chrono::high_resolution_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << ":::::::::::: Performance CPU-only code ::::::::::::" << std::endl;
  std::cout << "sequential result took " << t.count() << "ns." << std::endl;
  std::cout << "Distortion: " << d1 << std::endl;

  /*
    CUDA-Accelerated distortion
  */
  float d2 = 0;
  float* device_error_matrix;
  float* device_codebook;
  float* device_training_seq;
  float* device_reduce_sums;
  unsigned int* device_cells;
  checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cells, TRAINING_SIZE*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &device_training_seq, TRAINING_SIZE*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_reduce_sums, TRAINING_SIZE*sizeof(float)));

  checkCudaErrors(cudaMemcpy(device_training_seq, training_sequence, TRAINING_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cells, cells, TRAINING_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice));
  start = std::chrono::high_resolution_clock::now();
  dim3 grid_size = {TRAINING_SIZE / WARP_SIZE, 1, 1};
  dim3 block_size = {WARP_SIZE, 1, 1};
  distortion_gather<levels><<<grid_size, block_size>>>(device_training_seq, device_codebook, device_error_matrix, device_cells, device_reduce_sums);
  d2 = distortion_reduce(device_reduce_sums);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << ":::::::::::: Performance GPU-only code ::::::::::::" << std::endl;
  std::cout << "CUDA result took " << t.count() << "ns." << std::endl;
  std::cout << "Distortion: " << d2 << std::endl;
  checkCudaErrors(cudaFree(device_training_seq));
  checkCudaErrors(cudaFree(device_error_matrix));
  checkCudaErrors(cudaFree(device_codebook));
  checkCudaErrors(cudaFree(device_cells));
  free(cells);
  free(codebook);
  free(training_sequence);
  free(error_matrix);
}