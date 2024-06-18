#include "cosq.h"
#include <stdlib.h>
#include <float.h>
#include "spdlog/spdlog.h"
#include "cuda/nnc.cuh"
#include "cuda/cc.cuh"
#include "cuda/dist.cuh"
#include "cuda/nvidia.cuh"
#include "ext.h"

#define MAX_ERROR 0.0000001

/**
 * Host variables
 */
double* COSQ::training_sequence;
unsigned int COSQ::training_size;
unsigned int COSQ::levels;
unsigned int COSQ::bit_rate;
double* COSQ::error_matrix;
double* COSQ::q_points;


/**
 * Device variables
 */
// General
double* COSQ::Device::training_sequence;
double* COSQ::Device::error_matrix;
double* COSQ::Device::q_points;
// NNC
dim3 COSQ::Device::nnc_grid_size;
dim3 COSQ::Device::nnc_block_size;
unsigned int COSQ::Device::nnc_smem_size;
unsigned int* COSQ::Device::q_cells;
// CC
dim3 COSQ::Device::cc_grid_size;
dim3 COSQ::Device::cc_block_size;
unsigned int* COSQ::Device::cc_cardinality;
double* COSQ::Device::cc_cell_sums;
// Distortion
dim3 COSQ::Device::dist_grid_size;
dim3 COSQ::Device::dist_block_size;
unsigned int COSQ::Device::dist_smem_size;
double* COSQ::Device::reduction_sums;


/**
 * Allocate memory for device arrays.
 */
void COSQ::Device::init(double* training_sequence_, const unsigned int* training_size, double* error_matrix_, const unsigned int* levels) {
  // Memory allocation
  checkCudaErrors(cudaMalloc((void **) &training_sequence, (*training_size)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &error_matrix, (*levels)*(*levels)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &q_points, (*levels)*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &q_cells, (*training_size)*sizeof(unsigned int)));

  checkCudaErrors(cudaMalloc((void **) &cc_cardinality, (*levels)*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &cc_cell_sums, (*levels)*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &reduction_sums, (*training_size)*sizeof(double)));

  // Memory copying
  checkCudaErrors(cudaMemcpy(training_sequence, training_sequence_,
                            (*training_size)*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(error_matrix, error_matrix_,
                          (*levels)*(*levels)*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(cc_cardinality, 0, (*levels)*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(cc_cell_sums, 0, (*levels)*sizeof(double)));

  // CUDA kernel launch params
  nnc_grid_size = {*training_size, 1, 1};
  nnc_block_size = {WARP_SIZE, 1, 1};
  nnc_smem_size = 2 * (*levels) * sizeof(double);

  cc_grid_size = {*levels, 1, 1};
  cc_block_size = {WARP_SIZE, 1, 1};

  dist_grid_size = {*training_size / WARP_SIZE, 1, 1};
  dist_block_size = {WARP_SIZE, 1, 1};
  dist_smem_size = (*levels) * sizeof(double);
}

/**
 *
 */
void COSQ::init(double* training_sequence_, const unsigned int* training_size_) {
  COSQ::training_sequence = training_sequence_;
  COSQ::training_size = *training_size_;
  COSQ::error_matrix = (double*) malloc((COSQ::levels)*(COSQ::levels)*sizeof(double));
  if(COSQ::error_matrix == nullptr) {
    spdlog::error("Memory Allocation error: Failed to allocate memory for error_matrix!");
    return;
  }
  compute_error_matrix();
  COSQ::q_points = (double*) malloc((COSQ::levels)*sizeof(double));
  if(COSQ::q_points == nullptr) {
    spdlog::error("Memory Allocation error: Failed to allocate memory for q_points!");
    return;
  }
  Device::init(training_sequence_, training_size_, error_matrix, &levels);
}

void COSQ::finish() {
  free(COSQ::error_matrix);
  Device::finish();
}


/**
 * Free all memory on device.
 */
void COSQ::Device::finish() {
  checkCudaErrors(cudaFree(training_sequence));
  checkCudaErrors(cudaFree(error_matrix));
  checkCudaErrors(cudaFree(q_points));
  checkCudaErrors(cudaFree(q_cells));
  checkCudaErrors(cudaFree(cc_cardinality));
  checkCudaErrors(cudaFree(cc_cell_sums));
  checkCudaErrors(cudaFree(reduction_sums));
}



/**
 *
 */
inline double COSQ::polya_urn_error(int j, int i, int num_bits) {
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
 * TODO: Use CUDA to accelerate this
 */
void COSQ::compute_error_matrix() {
  for(int i = 0; i < (levels); i++) {
    for(int j = 0; j < (levels); j++) {
      error_matrix[j + i * (levels)] = polya_urn_error(j, i, bit_rate);
    }
  }
}

// /**
//  * TODO
//  */
// void COSQ::split() {
//   const double delta = 0.01;
//   double* temp = (double*) malloc(sizeof(double) * (levels));
//   free(temp);
// }

// /**
//  * TODO
//  */
// void COSQ::sim_annealing() {}

void nnc_cpu(unsigned int training_size, unsigned int* cells, double* training_sequence, double* codebook, int levels, double* error_matrix,
    double* cell_sums, double* cc_sums, unsigned int* cc_cardinality) {
  double min = __FLT_MAX__;
  int min_index = -1;
  double sum = 0;
  double c = 0;
  for(int i = 0; i < training_size; i++) {
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

void cc_cpu(int levels, double* error_matrix, double* cc_sums, unsigned int* cc_cardinality, double* codebook) {
  double numerator = 0;
  double denominator = 0;
  for (int j = 0; j < levels; j++) {
      for (int i = 0; i < levels; i++) {
          numerator += error_matrix[j + levels * i] * cc_sums[i];
      }
      for (int i = 0; i < levels; i++) {
          denominator += error_matrix[j + levels * i] * cc_cardinality[i];
      }
      codebook[j] = numerator / denominator;
      numerator = 0;
      denominator = 0;
  }
}

double distortion_cpu(unsigned int training_size, unsigned int levels, double* training_sequence, double* error_matrix, double* codebook, unsigned int* cells) {
  double d = 0;
  double c = 0;
  for(int i = 0; i < training_size; i++) {
    for(int j = 0; j < levels; j++) {
      double y = error_matrix[j + levels*cells[i]] * (training_sequence[i] - codebook[j]) * (training_sequence[i] - codebook[j]) - c;
      double t = d + y;
      c = (t - d) - y;
      d = t;
    }
  }
  return d / training_size;
}

void nnc_cc_test(unsigned int training_size, unsigned int levels, double* training_sequence, unsigned int* cpu_cells,
                 unsigned int* cuda_cc_cardinality, double* cuda_cc_training_sums) {
  // First, take cpu cells and compute training sequence sums and count cardinality.
  double cc_training_sums[levels] = {};
  unsigned int cardinality[levels] = {};
  unsigned int idx;
  double c[levels] = {};
  double sum;
  for(int i = 0; i < training_size; i++) {
    idx = cpu_cells[i];
    cardinality[idx]++;
    // Kahan summation on cc_training_sums[idx].
    sum = cc_training_sums[idx];
    double y = training_sequence[i] - c[idx];
    double t = sum + y;
    c[idx] = (t - sum) - y;
    cc_training_sums[idx] = t;
  }
  bool equal = true;
  spdlog::info("Executing test cell cardinality and min sums...");
  for(int i = 0; i < levels; i++) {
    if((cuda_cc_cardinality[i] != cardinality[i]) || abs(cc_training_sums[i] - cuda_cc_training_sums[i]) > MAX_ERROR) {
      spdlog::error("Codebook element: {:d}. Cardinality of CUDA {:d} vs. CPU {:d} ", i, cuda_cc_cardinality[i], cardinality[i]);
      spdlog::error("Codebook element: {:d}. cc sum of CUDA {:f} vs. cc sum CPU {:f} ", i, cuda_cc_training_sums[i], cc_training_sums[i]);
      spdlog::error("NNC CC TEST failed!");
      equal = false;
      break;
    }
  }
  if(equal)
    spdlog::info("NNC CC TEST PASSED");
}

void nnc_cells_test(unsigned int training_size, unsigned int levels, unsigned int* cuda_cells, unsigned int* cpu_cells, double* nnc_sums) {
  bool equal = true;
  spdlog::info("Executing test on cells...");
  for(int i = 0; i < training_size; i++) {
    if((cuda_cells[i] != cpu_cells[i]) && (nnc_sums[i*levels + cuda_cells[i]] != nnc_sums[i*levels + cpu_cells[i]])) {
      spdlog::error("Training element: {:d}. CUDA {:d} vs. CPU {:d} ", i, cuda_cells[i], cpu_cells[i]);
      spdlog::error("Training element: {:d}. CUDA min sum {:f} vs. CPU min sum {:f} ", i, nnc_sums[i*levels + cuda_cells[i]], nnc_sums[i*levels + cpu_cells[i]]);
      spdlog::info("NNC CELLS TEST FAILED");
      equal = false;
      break;
    }
  }
  if(equal)
    spdlog::info("NNC CELLS TEST PASSED");
}

void cc_correct(double* codebook_seq, double* codebook_cuda, unsigned int levels) {
  spdlog::info("Performing correctness test CC");
    bool correct = true;
    for (int i = 0; i < levels; i++) {
      if (fabsf64(codebook_seq[i] - codebook_cuda[i]) > MAX_ERROR) {
          spdlog::error("The codebooks DO NOT match!\n");
          spdlog::error("Disagreement at {:d}: codebook_seq {:f}, codebook gpu {:f}", i, codebook_seq[i], codebook_cuda[i]);
          correct = false;
          break;
      }
    }
    if (correct)
      spdlog::info("The codebooks match! CC Correctness test passed!\n");
}

/**
 *
 */
double* COSQ::train(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate) {
  double dist_prev = DBL_MAX, dist_curr = 0;
  COSQ::levels = 1 << *bit_rate;
  COSQ::bit_rate = *bit_rate;
  init(training_sequence, training_size);
  // For now, just use first few training seq elements
  for(int i = 0; i < levels; i++)
    q_points[i] = training_sequence[i];
  checkCudaErrors(cudaMemcpy(COSQ::Device::q_points, COSQ::q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  // Testing data
  unsigned int* cpu_cells = (unsigned int*) malloc(sizeof(unsigned int) * *training_size);
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * *training_size);
  double* all_sums_nnc = (double*) malloc(sizeof(double) * *training_size * levels);
  double* cpu_cc_cell_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cpu_cc_cardinal = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_cc_training_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cuda_cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_codebook = (double*) malloc(sizeof(double) * levels);
  memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
  memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);

  // COSQ algorithm
  while(true) {
    nnc<<<COSQ::Device::nnc_grid_size, COSQ::Device::nnc_block_size, COSQ::Device::nnc_smem_size>>>(levels, COSQ::Device::training_sequence, COSQ::Device::q_points,
        COSQ::Device::error_matrix, COSQ::Device::q_cells, COSQ::Device::cc_cell_sums, COSQ::Device::cc_cardinality);
    nnc_cpu(*training_size, cpu_cells, training_sequence, COSQ::q_points, levels, COSQ::error_matrix, all_sums_nnc, cpu_cc_cell_sums, cpu_cc_cardinal);
    // NNC CHECK!
    checkCudaErrors(cudaMemcpy(cuda_cells, COSQ::Device::q_cells, sizeof(unsigned int) * *training_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, COSQ::Device::cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, COSQ::Device::cc_cell_sums, levels*sizeof(double), cudaMemcpyDeviceToHost));
    nnc_cells_test(*training_size, levels, cuda_cells, cpu_cells, all_sums_nnc);
    nnc_cc_test(*training_size, levels, training_sequence, cpu_cells, cuda_cc_cardinality, cuda_cc_training_sums);
    cc<<<COSQ::Device::cc_grid_size, COSQ::Device::cc_block_size>>>(levels, COSQ::Device::q_points, COSQ::Device::error_matrix,
        COSQ::Device::cc_cell_sums, COSQ::Device::cc_cardinality);
    cc_cpu(levels, error_matrix, cpu_cc_cell_sums, cpu_cc_cardinal, COSQ::q_points);
    checkCudaErrors(cudaMemcpy(cuda_codebook, COSQ::Device::q_points, levels*sizeof(double), cudaMemcpyDeviceToHost));
    cc_correct(COSQ::q_points, cuda_codebook, levels);
    distortion_gather<<<COSQ::Device::dist_grid_size, COSQ::Device::dist_block_size, COSQ::Device::dist_smem_size>>>(levels, COSQ::Device::training_sequence,
        COSQ::Device::q_points, COSQ::Device::error_matrix, COSQ::Device::q_cells, COSQ::Device::reduction_sums);
    dist_curr = distortion_reduce(COSQ::training_size, COSQ::Device::reduction_sums);
    double d_cpu = distortion_cpu(*training_size, levels, training_sequence, error_matrix, COSQ::q_points, cpu_cells);
    if(fabsf64(d_cpu - dist_curr) > MAX_ERROR) {
      spdlog::error("Distortion test failed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    } else {
      spdlog::info("Distortion test passed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    }
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      return nullptr;
    }
    dist_prev = dist_curr;
    checkCudaErrors(cudaMemset(COSQ::Device::cc_cardinality, 0, (levels)*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(COSQ::Device::cc_cell_sums, 0, (levels)*sizeof(double)));
    memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
    memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);
  }
  free(cpu_cells);
  free(cuda_cells);
  free(cuda_codebook);
  free(all_sums_nnc);
  free(cpu_cc_cell_sums);
  free(cpu_cc_cardinal);
  free(cuda_cc_training_sums);
  free(cuda_cc_cardinality);
}
