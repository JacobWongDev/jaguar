#include "cosq.h"
#include <stdlib.h>
#include <float.h>
#include "spdlog/spdlog.h"
#include "cuda/nnc.cuh"
#include "cuda/cc.cuh"
#include "cuda/dist.cuh"
#include "cuda/nvidia.cuh"
#include "ext.h"

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

/**
 *
 */
double* COSQ::train(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate) {
  double dist_prev = DBL_MAX, dist_curr = 0;
  COSQ::levels = 1 << *bit_rate;
  COSQ::bit_rate = *bit_rate;
  init(training_sequence, training_size);
  // sim_annealing(Host::q_points, training_sequence, training_size, error_matrix, bit_rate, &levels);
  // For now, just use first few training seq elements
  for(int i = 0; i < levels; i++)
    q_points[i] = training_sequence[i];
  checkCudaErrors(cudaMemcpy(COSQ::Device::q_points, COSQ::q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  // COSQ algorithm
  while(true) {
    nnc<<<COSQ::Device::nnc_grid_size, COSQ::Device::nnc_block_size, COSQ::Device::nnc_smem_size>>>(levels, COSQ::Device::training_sequence, COSQ::Device::q_points,
        COSQ::Device::error_matrix, COSQ::Device::q_cells, COSQ::Device::cc_cell_sums, COSQ::Device::cc_cardinality);
    cc<<<COSQ::Device::cc_grid_size, COSQ::Device::cc_block_size>>>(levels, COSQ::Device::q_points, COSQ::Device::error_matrix,
        COSQ::Device::cc_cell_sums, COSQ::Device::cc_cardinality);
    distortion_gather<<<COSQ::Device::dist_grid_size, COSQ::Device::dist_block_size, COSQ::Device::dist_smem_size>>>(levels, COSQ::Device::training_sequence,
        COSQ::Device::q_points, COSQ::Device::error_matrix, COSQ::Device::q_cells, COSQ::Device::reduction_sums);
    dist_curr = distortion_reduce(COSQ::training_size, COSQ::Device::reduction_sums);
    spdlog::info("Distortion is {:f}", dist_curr);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
      // TODO: Return copy of COSQ::q_points
    }
    dist_prev = dist_curr;
    checkCudaErrors(cudaMemset(COSQ::Device::cc_cardinality, 0, (levels)*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(COSQ::Device::cc_cell_sums, 0, (levels)*sizeof(double)));
  }
}
