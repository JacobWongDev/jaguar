#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "cosq.cuh"

#define POLYA_EPSILON 0.01
#define POLYA_DELTA 0
#define MAX_ERROR 0.0000001
#define ITER 10

void check(cudaError_t error, const char* file, int line) {
    if(cudaSuccess != error) {
        printf("CUDA error in %s: line %d code=%d(%s): %s\n", file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
    }
}

#define checkCudaErrors(error) check(error, __FILE__, __LINE__);

void nnc_cpu(unsigned int* cells, double* training_sequence, double* codebook, int levels, double* error_matrix,
    double* cell_sums, double* cc_sums, unsigned int* cc_cardinality) {
  double min = __FLT_MAX__;
  int min_index = -1;
  double sum = 0;
  double c = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    double target = training_sequence[i];
    for(int l = 0; l < levels; l++) {
      // Kahan summation
      for(int j = 0; j < levels; j++) {
        double y = error_matrix[levels*l + j] * (target - codebook[j]) * (target - codebook[j]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
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
}

inline double polya_urn_error(int j, int i, int num_bits) {
  double temp;
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

/**
 * Computes channel transition matrix p(j|i) where
 * i is the input symbol
 * j is the output symbol
 *
 * To promote coalesced memory access on the GPU, the matrix
 * is calculated in transposed form
 *
 * Typical: p(j|i) = mat[j + n*i]
 * 
 * Transposed access: p(j|i) = mat[i + n*j]
 *
 */
double* compute_error_matrix(unsigned int levels, unsigned int rate) {
  double* error_matrix = (double*) malloc(sizeof(double) * levels * levels);
  for(int i = 0; i < levels; i++) {
      for(int j = 0; j < levels; j++) {
          error_matrix[i + j * levels] = polya_urn_error(j, i, rate);
      }
  }
  return error_matrix;
}

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
double* generate_normal_sequence() {
  double* normal_sequence = (double*) malloc(TRAINING_SIZE * sizeof(double));
  std::default_random_engine rng;
  rng.seed(31);
  std::normal_distribution<double> distribution(10, 1);
  for(int i = 0; i < TRAINING_SIZE; i++) {
      normal_sequence[i] = distribution(rng);
  }
  return normal_sequence;
}

void nnc_cc_test(unsigned int levels, double* training_sequence, unsigned int* cuda_cells,
                 unsigned int* cuda_cc_cardinality, double* cuda_cc_training_sums) {
  // First, take cpu cells and compute training sequence sums and count cardinality.
  double cc_training_sums[levels] = {};
  unsigned int cardinality[levels] = {};
  unsigned int idx;
  double c[levels] = {};
  double sum;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    idx = cuda_cells[i];
    cardinality[idx]++;
    // Kahan summation on cc_training_sums[idx].
    sum = cc_training_sums[idx];
    double y = training_sequence[i] - c[idx];
    double t = sum + y;
    c[idx] = (t - sum) - y;
    cc_training_sums[idx] = t;
  }
  bool equal = true;
  std::cout << "Executing test cell cardinality and min sums..." << std::endl;
  for(int i = 0; i < levels; i++) {
    if((cuda_cc_cardinality[i] != cardinality[i]) || abs(cc_training_sums[i] - cuda_cc_training_sums[i]) > MAX_ERROR) {
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

void nnc_cells_test(unsigned int levels, unsigned int* cuda_cells, unsigned int* cpu_cells, double* nnc_sums) {
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
  unsigned int rate = 4;
  const unsigned int levels = 1 << rate;
  double* training_sequence = generate_normal_sequence();
  double* error_matrix = compute_error_matrix(levels, rate);
  double* codebook = (double*) malloc(sizeof(double) * levels);
  double* cc_training_sums = (double*) calloc(levels, sizeof(double));
  unsigned int* cc_cardinality = (unsigned int*) calloc(levels, sizeof(unsigned int));
  double* nnc_sums = (double*) malloc(sizeof(double) * levels * TRAINING_SIZE); // can delete later, used for testing!
  // intialize codebook to first <levels> training samples
  for(int i = 0; i < levels; i++)
    codebook[i] = training_sequence[i];

  /*
    Sequential NNC
  */
  std::cout << ":::::::::::: Performance CPU-only code ::::::::::::" << std::endl;
  std::chrono::_V2::system_clock::time_point start, end;
  std::chrono::milliseconds exec_time;
  start = std::chrono::high_resolution_clock::now();
  unsigned int* cpu_cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  nnc_cpu(cpu_cells, training_sequence, codebook, levels, error_matrix, nnc_sums, cc_training_sums, cc_cardinality);
  end = std::chrono::high_resolution_clock::now();
  exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "The exec time is " << exec_time.count() << "ms." << std::endl;

  /*
    CUDA-Accelerated NNC
  */
  double* device_training_seq;
  // double* device_error_matrix;
  double* device_codebook;
  double* device_cc_training_sums;
  unsigned int* device_cc_cardinality;
  unsigned int* device_cells;
  checkCudaErrors(cudaMalloc((void **) &device_training_seq, TRAINING_SIZE*sizeof(double)));
  // checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &device_cc_training_sums, levels*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &device_cc_cardinality, levels*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &device_cells, TRAINING_SIZE*sizeof(double)));
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  double* cuda_cc_training_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cuda_cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);

  checkCudaErrors(cudaMemcpy(device_training_seq, training_sequence, TRAINING_SIZE*sizeof(double), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_codebook, codebook, levels*sizeof(double), cudaMemcpyHostToDevice));

 /*
    Kernel nnc1
  */
  std::cout << ":::::::::::: Performance nnc1 ::::::::::::" << std::endl;
  unsigned int sum = 0;
  unsigned int smem_size;
  checkCudaErrors(cudaMemcpyToSymbol(c_q_points, codebook, levels*sizeof(double)));
  checkCudaErrors(cudaMemcpyToSymbol(tm, error_matrix, levels*levels*sizeof(double)));
  for(int i = 0; i < ITER; i++) {
    start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemset(device_cc_training_sums, 0, levels*sizeof(double)));
    checkCudaErrors(cudaMemset(device_cc_cardinality, 0, levels*sizeof(unsigned int)));
    dim3 block_size = {1024, 1, 1};
    dim3 grid_size = {TRAINING_SIZE / (block_size.x / levels), 1, 1};
    nnc1<<<grid_size, block_size>>>(levels, device_training_seq, device_cells);
    grid_size = {levels, 1, 1};
    block_size = {64, 1, 1};
    smem_size = (block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));
    cc_p1<<<grid_size, block_size, smem_size>>>(device_training_seq, device_cells, device_cc_training_sums, device_cc_cardinality);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(i == 0) {
      std::cout << "Warm-up time is " << exec_time.count() << "ms." << std::endl;
    } else {
      sum += exec_time.count();
    }
  }
  checkCudaErrors(cudaMemcpy(cuda_cells, device_cells, TRAINING_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, device_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, device_cc_training_sums, levels*sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "The average of the remaining exec times is " << sum / (ITER - 1) << "ms." << std::endl;
  std::cout << ":::::::::::: Tests for nnc1 ::::::::::::" << std::endl;
  nnc_cells_test(levels, cuda_cells, cpu_cells, nnc_sums);
  nnc_cc_test(levels, training_sequence, cuda_cells, cuda_cc_cardinality, cuda_cc_training_sums);

  checkCudaErrors(cudaFree(device_training_seq));
  // checkCudaErrors(cudaFree(device_error_matrix));
  checkCudaErrors(cudaFree(device_codebook));
  checkCudaErrors(cudaFree(device_cc_cardinality));
  checkCudaErrors(cudaFree(device_cc_training_sums));
  checkCudaErrors(cudaFree(device_cells));
  free(cpu_cells);
  free(nnc_sums);
  free(cc_cardinality);
  free(cc_training_sums);
  free(cuda_cc_cardinality);
  free(cuda_cc_training_sums);
  free(cuda_cells);
  free(codebook);
  free(training_sequence);
  free(error_matrix);
}