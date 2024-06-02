#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "cosq.cuh"

#define TRAINING_SIZE 1048576
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

unsigned int* nnc_cpu(float* training_sequence, float* codebook, int levels, float* error_matrix,
                      float* cell_sums, float* cc_sums, unsigned int* cc_cardinality) {
  float min = __FLT_MAX__;
  unsigned int* cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  int min_index = -1;
  float sum = 0;
  float c = 0;
  float next, next_sum;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    float target = training_sequence[i];
    for(int l = 0; l < levels; l++) {
      // Kahan summation
      for(int j = 0; j < levels; j++) {
        next = error_matrix[levels*l + j] * (target - codebook[j]) * (target - codebook[j]) + c;
        next_sum = sum + next;
        c = next - (next_sum - sum);
        sum = next_sum;
      }
      if(sum < min) {
        min_index = l;
        min = sum;
      }
      cell_sums[levels*i + l] = sum;
      sum=0;
      c=0;
    }
    cells[i] = min_index;
    // For Centroid Condition
    cc_cardinality[min_index]++; // update count
    cc_sums[min_index] += target; // running sum
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
  std::normal_distribution<float> distribution(10, 1);
  for(int i = 0; i < TRAINING_SIZE; i++) {
      normal_sequence[i] = distribution(rng);
  }
  return normal_sequence;
}

void nnc_cc_test(unsigned int levels, float* training_sequence, unsigned int* cuda_cells,
                 unsigned int* cuda_cc_cardinality, float* cuda_cc_training_sums) {
  // First, take cpu cells and compute training sequence sums and count cardinality.
  float cc_training_sums[levels] = {};
  unsigned int cardinality[levels] = {};
  unsigned int idx;
  float c[levels] = {};
  float next, next_sum;
  float sum;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    idx = cuda_cells[i];
    cardinality[idx]++;
    // Kahan summation on cc_training_sums[idx].
    sum = cc_training_sums[idx];
    next = training_sequence[i] + c[idx];
    next_sum = sum + next;
    c[idx] = next - (next_sum - sum);
    cc_training_sums[idx] = next_sum;
  }
  bool equal = true;
  std::cout << "Executing test cell cardinality and min sums..." << std::endl;
  for(int i = 0; i < levels; i++) {
    if((cuda_cc_cardinality[i] != cardinality[i]) || abs(cc_training_sums[i] - cuda_cc_training_sums[i]) > FLOAT_ERROR) {
      std::cout << "Codebook element: " << i << " cuda nnc cardinality: " << cardinality[i] << " cuda cc cardinality: "
                << cuda_cc_cardinality[i] << std::endl;
      std::cout << "Codebook element: " << i << std::setprecision(10) << " cuda nnc sum: " << cc_training_sums[i]
                << " cuda cc sum: " << cuda_cc_training_sums[i] << std::endl;
      std::cout << "Test failed!" << std::endl;
      equal = false;
      break;
    }
  }
  if(equal)
    std::cout << "Test Passed!" << std::endl;
}

void nnc_cells_test(unsigned int levels, unsigned int* cuda_cells, unsigned int* cpu_cells, float* nnc_sums) {
  bool equal = true;
  std::cout << "Executing test on cells..." << std::endl;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    if((cuda_cells[i] != cpu_cells[i]) && (nnc_sums[i*levels + cuda_cells[i]] != nnc_sums[i*levels + cpu_cells[i]])) {
      std::cout << "Training element: " << i << " cpu: " << cpu_cells[i] << " cuda: " << cuda_cells[i] << std::endl;
      std::cout << "Training element: " << i << std::setprecision(10) << " cpu min sum: " << nnc_sums[i*levels + cpu_cells[i]]
                << " cuda min sum: " << nnc_sums[i*levels + cuda_cells[i]] << std::endl;
      std::cout << "Test failed!" << std::endl;
      equal = false;
      break;
    }
  }
  if(equal)
    std::cout << "Test Passed!" << std::endl;
}

int main(int argc, char** argv) {
  const unsigned int levels = 1 << RATE;
  float* training_sequence = generate_normal_sequence();
  float* error_matrix = compute_error_matrix(levels);
  float* codebook = (float*) malloc(sizeof(float) * levels);
  float* cc_training_sums = (float*) calloc(levels, sizeof(float));
  unsigned int* cc_cardinality = (unsigned int*) calloc(levels, sizeof(unsigned int));
  float* zero_cc_training_sums = (float*) calloc(levels, sizeof(float));
  unsigned int* zero_cc_cardinality = (unsigned int*) calloc(levels, sizeof(unsigned int));
  float* nnc_sums = (float*) malloc(sizeof(float) * levels * TRAINING_SIZE); // can delete later, used for testing!
  // intialize codebook to first <levels> training samples
  for(int i = 0; i < levels; i++)
    codebook[i] = training_sequence[i];

  /*
    Sequential NNC
  */
  auto start = std::chrono::high_resolution_clock::now();
  unsigned int* cpu_cells = nnc_cpu(training_sequence, codebook, levels, error_matrix, nnc_sums, cc_training_sums, cc_cardinality);
  auto end = std::chrono::high_resolution_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << ":::::::::::: Performance CPU-only code ::::::::::::" << std::endl;
  std::cout << "sequential result took " << t.count() << "ms." << std::endl;

  /*
    CUDA-Accelerated NNC
  */
  float* device_training_seq;
  float* device_error_matrix;
  float* device_codebook;
  float* device_cc_training_sums;
  unsigned int* device_cc_cardinality;
  unsigned int* device_cells;
  unsigned int* device_cells_2;
  checkCudaErrors(cudaMalloc((void **) &device_training_seq, TRAINING_SIZE*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cc_training_sums, levels*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cc_cardinality, levels*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &device_cells, TRAINING_SIZE*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &device_cells_2, TRAINING_SIZE*sizeof(float)));
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  float* cuda_cc_training_sums = (float*) malloc(sizeof(float) * levels);
  unsigned int* cuda_cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);

  /*
    Kernel nnc_e32
  */
  start = std::chrono::high_resolution_clock::now();
  checkCudaErrors(cudaMemcpy(device_training_seq, training_sequence, TRAINING_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cc_training_sums, zero_cc_training_sums, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cc_cardinality, zero_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyHostToDevice));
  dim3 grid_size = {TRAINING_SIZE, 1, 1};
  dim3 block_size = {32, 1, 1};
  nnc_e32<levels><<<grid_size, block_size>>>(device_training_seq, device_codebook, device_error_matrix,
                                             device_cells, device_cc_training_sums, device_cc_cardinality);
  checkCudaErrors(cudaMemcpy(cuda_cells, device_cells, TRAINING_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, device_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, device_cc_training_sums, levels*sizeof(float), cudaMemcpyDeviceToHost));
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << ":::::::::::: Performance nnc_e32 ::::::::::::" << std::endl;
  std::cout << "Kernel nnc_e32 result took " << t.count() << "ms." << std::endl;
  std::cout << ":::::::::::: Tests for nnc_e32 ::::::::::::" << std::endl;
  nnc_cells_test(levels, cuda_cells, cpu_cells, nnc_sums);
  nnc_cc_test(levels, training_sequence, cuda_cells, cuda_cc_cardinality, cuda_cc_training_sums);

  /*
    Kernel nnc_e32_v2
  */
  start = std::chrono::high_resolution_clock::now();
  checkCudaErrors(cudaMemcpy(device_training_seq, training_sequence, TRAINING_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cc_training_sums, zero_cc_training_sums, levels*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_cc_cardinality, zero_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyHostToDevice));
  grid_size = {TRAINING_SIZE, 1, 1};
  block_size = {32, 1, 1};
  nnc_e32_v2<levels><<<grid_size, block_size>>>(device_training_seq, device_codebook, device_error_matrix,
                                                device_cells_2, device_cc_training_sums, device_cc_cardinality);
  checkCudaErrors(cudaMemcpy(cuda_cells, device_cells_2, TRAINING_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, device_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, device_cc_training_sums, levels*sizeof(float), cudaMemcpyDeviceToHost));
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << ":::::::::::: Performance nnc_e32_v2 ::::::::::::" << std::endl;
  std::cout << "Kernel nnc_e32_v2 result took " << t.count() << "ms." << std::endl;
  std::cout << ":::::::::::: Tests for nnc_e32_v2 ::::::::::::" << std::endl;
  nnc_cells_test(levels, cuda_cells, cpu_cells, nnc_sums);
  nnc_cc_test(levels, training_sequence, cuda_cells, cuda_cc_cardinality, cuda_cc_training_sums);

  checkCudaErrors(cudaFree(device_training_seq));
  checkCudaErrors(cudaFree(device_error_matrix));
  checkCudaErrors(cudaFree(device_codebook));
  checkCudaErrors(cudaFree(device_cc_cardinality));
  checkCudaErrors(cudaFree(device_cc_training_sums));
  checkCudaErrors(cudaFree(device_cells));
  checkCudaErrors(cudaFree(device_cells_2));
  free(cpu_cells);
  free(nnc_sums);
  free(zero_cc_cardinality);
  free(zero_cc_training_sums);
  free(cc_cardinality);
  free(cc_training_sums);
  free(cuda_cc_cardinality);
  free(cuda_cc_training_sums);
  free(cuda_cells);
  free(codebook);
  free(training_sequence);
  free(error_matrix);
}