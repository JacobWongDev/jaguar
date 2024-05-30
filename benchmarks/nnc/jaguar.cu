#include <random>
#include <chrono>
#include <iostream>
#include "cosq.cuh"

#define TRAINING_SIZE 1048576
#define RATE 8
#define POLYA_EPSILON 0
#define POLYA_DELTA 0

void check(cudaError_t error, const char* file, int line) {
    if(cudaSuccess != error) {
        printf("CUDA error in %s: line %d code=%d(%s): %s\n", file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
    }
}

#define checkCudaErrors(error) check(error, __FILE__, __LINE__);

unsigned int* nearest_neighbour(float* training_sequence, float* codebook, int levels, float* error_matrix, float* cell_sums) {
  float min = __FLT_MAX__;
  unsigned int* cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  int min_index = -1;
  float sum = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    float target = training_sequence[i];
    for(int l = 0; l < levels; l++) {
      for(int j = 0; j < levels; j++) {
        sum += error_matrix[levels*l + j] * (target - codebook[j]) * (target - codebook[j]);
      }
      if(sum < min) {
        min_index = l;
        min = sum;
      }
      sum=0;
    }
    cells[i] = min_index;
    cell_sums[i] = min;
    sum = 0;
    min_index = -1;
    min = __FLT_MAX__;
  }
  return cells;
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
  std::normal_distribution<float> distribution(0, 1);
  for(int i = 0; i < TRAINING_SIZE; i++) {
      normal_sequence[i] = distribution(rng);
  }
  return normal_sequence;
}

int main(int argc, char** argv) {
  const unsigned int levels = 1 << RATE;
  float* normal_sequence = generate_normal_sequence();
  float* error_matrix = compute_error_matrix(levels);
  float* codebook = (float*) malloc(sizeof(float) * levels);
  float* cell_sums = (float*) malloc(sizeof(float) * TRAINING_SIZE); // can delete later, used for testing!
  for(int i = 0; i < levels; i++)
      codebook[i] = normal_sequence[i];
  // intialize codebook to first training samples
  auto start = std::chrono::high_resolution_clock::now();
  unsigned int* seq_cells = nearest_neighbour(normal_sequence, codebook, levels, error_matrix, cell_sums);
  auto end = std::chrono::high_resolution_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "sequential result took " << t.count() << "ms." << std::endl;

  /*
    CUDA-Accelerated NNC
  */
  float* device_training_seq;
  float* device_error_matrix;
  float* device_codebook;
  unsigned int* device_cells;
  checkCudaErrors(cudaMalloc((void **) &device_training_seq, TRAINING_SIZE*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cells, TRAINING_SIZE*sizeof(float)));
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);

  /*
    Kernel nnc_e32
  */
  start = std::chrono::high_resolution_clock::now();
  checkCudaErrors(cudaMemcpy(device_training_seq, normal_sequence, TRAINING_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid_size = {TRAINING_SIZE, 1, 1};
  dim3 block_size = {32, 1, 1};
  nnc_e32<levels><<<grid_size, block_size>>>(device_training_seq, device_codebook, device_error_matrix, device_cells);
  checkCudaErrors(cudaMemcpy(cuda_cells, device_cells, TRAINING_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Kernel nnc_e32 result took " << t.count() << "ms." << std::endl;

  // compare results
  bool equal = true;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    if((cuda_cells[i] != seq_cells[i]) && (cell_sums[cuda_cells[i]] != cell_sums[seq_cells[i]])) {
      std::cout << "Iteration: " << i << " seq: " << seq_cells[i] << " cuda: " << cuda_cells[i] << std::endl;
      std::cout << "Not equal! Test failed!" << std::endl;
      equal = false;
      break;
    }
  }
  if(equal)
    std::cout << "Equal! Test Passed!" << std::endl;

  checkCudaErrors(cudaFree(device_training_seq));
  checkCudaErrors(cudaFree(device_error_matrix));
  checkCudaErrors(cudaFree(device_codebook));
  checkCudaErrors(cudaFree(device_cells));
  free(seq_cells);
  free(cell_sums);
  free(cuda_cells);
  free(codebook);
  free(normal_sequence);
  free(error_matrix);
}