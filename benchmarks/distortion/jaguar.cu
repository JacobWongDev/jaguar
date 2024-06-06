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

float distortion_cpu(unsigned int levels, float* training_sequence, float* error_matrix, float* codebook, unsigned int* cells) {
  float d = 0;
  float d_local = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    for(int j = 0; j < levels; j++) {
      d_local += error_matrix[j + levels*cells[i]] * (training_sequence[i] - codebook[j]) * (training_sequence[i] - codebook[j]);
    }
    d += d_local;
    d_local = 0;
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
  float* d2 = 0;
  float* device_error_matrix;
  float* device_codebook;
  float* device_training_seq;
  float* device_intermediate;
  unsigned int* device_cells;
  checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cells, TRAINING_SIZE*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &device_training_seq, TRAINING_SIZE*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_intermediate, TRAINING_SIZE*sizeof(float)));

  checkCudaErrors(cudaMemcpy(device_training_seq, training_sequence, TRAINING_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cells, cells, TRAINING_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice));
  start = std::chrono::high_resolution_clock::now();
  dim3 grid_size = {TRAINING_SIZE / WARP_SIZE, 1, 1};
  dim3 block_size = {WARP_SIZE, 1, 1};
  distortion_gather<levels><<<grid_size, block_size>>>(device_training_seq, device_codebook, device_error_matrix, device_cells, device_intermediate);
  // TODO: Need to write distortion_reduce to calculate distortion.
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