#include <stdlib.h>
#include <float.h>
#include "cosq.h"
#include "spdlog/spdlog.h"
#include "cuda/nnc.cuh"
#include "cuda/cc.cuh"
#include "cuda/dist.cuh"
#include "cuda/nvidia.cuh"
#include "ext.h"

Split::Split(COSQ* cosq, Device* device) {
  this->cosq = cosq;
  this->device = device;
}

void Split::split_lt32() {
  double* temp = NULL;
  double* s_codebook = (double*) malloc(sizeof(double) * cosq->levels);
  cosq->q_points = (double*) malloc(sizeof(double) * cosq->levels);
  // Compute centroid of training sequence
  double sum = 0;
  for(int i = 0; i < cosq->training_size; i++)
      sum += cosq->training_sequence[i];
  cosq->q_points[0] = sum / cosq->training_size;
  nnc_block_size = {WARP_SIZE, 1, 1};
  cc_cell_sums = (double*) malloc(sizeof(double) * cosq->levels);
  cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * cosq->levels);
  unsigned int rate = 0;
  unsigned int s_levels = 1;
  while(s_levels < cosq->levels) {
    // printArr(cosq->q_points, s_levels);
    for(int i = 0; i < s_levels; i++) {
      s_codebook[2*i] = cosq->q_points[i] - delta;
      s_codebook[2*i+1] = cosq->q_points[i] + delta;
    }
    temp = cosq->q_points;
    cosq->q_points = s_codebook;
    s_codebook = temp;
    s_levels <<= 1;
    rate++;
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, s_levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, s_levels*sizeof(double)));
    nnc_grid_size = {cosq->training_size * s_levels / WARP_SIZE, 1, 1};
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_error_matrix(cosq->error_matrix, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->error_matrix, cosq->error_matrix, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    s_nnc_lt32<<<nnc_grid_size, nnc_block_size>>>(s_levels, device->training_sequence, device->q_points,
        device->error_matrix, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cc_cell_sums, device->cc_cell_sums, sizeof(double) * s_levels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cc_cardinality, device->cc_cardinality, sizeof(unsigned int) * s_levels, cudaMemcpyDeviceToHost));
    cc_lt32(s_levels, cosq->error_matrix, cc_cell_sums, cc_cardinality, cosq->q_points);
  }
  free(s_codebook);
  free(cc_cell_sums);
  free(cc_cardinality);
}

void Split::split_ge32() {
  double* temp = NULL;
  double* s_codebook = (double*) malloc(sizeof(double) * cosq->levels);
  cosq->q_points = (double*) malloc(sizeof(double) * cosq->levels);
  cc_cell_sums = (double*) malloc(sizeof(double) * cosq->levels);
  cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * cosq->levels);
  // Compute centroid of training sequence
  double sum = 0;
  for(int i = 0; i < cosq->training_size; i++)
    sum += cosq->training_sequence[i];
  cosq->q_points[0] = sum / cosq->training_size;
  nnc_block_size = {WARP_SIZE, 1, 1};
  cc_block_size = {WARP_SIZE, 1, 1};
  unsigned int rate = 0;
  unsigned int s_levels = 1;
  while(s_levels < 32) {
    for(int i = 0; i < s_levels; i++) {
      s_codebook[2*i] = cosq->q_points[i] - delta;
      s_codebook[2*i+1] = cosq->q_points[i] + delta;
    }
    temp = cosq->q_points;
    cosq->q_points = s_codebook;
    s_codebook = temp;
    s_levels <<= 1;
    rate++;
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, s_levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, s_levels*sizeof(double)));
    nnc_grid_size = {cosq->training_size * s_levels / WARP_SIZE, 1, 1};
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_error_matrix(cosq->error_matrix, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->error_matrix, cosq->error_matrix, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    s_nnc_lt32<<<nnc_grid_size, nnc_block_size>>>(s_levels, device->training_sequence, device->q_points,
        device->error_matrix, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cc_cell_sums, device->cc_cell_sums, sizeof(double) * s_levels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cc_cardinality, device->cc_cardinality, sizeof(unsigned int) * s_levels, cudaMemcpyDeviceToHost));
    cc_lt32(s_levels, cosq->error_matrix, cc_cell_sums, cc_cardinality, cosq->q_points);
  }
  nnc_grid_size = {cosq->training_size, 1, 1};
  while(s_levels < cosq->levels) {
    for(int i = 0; i < s_levels; i++) {
      s_codebook[2*i] = cosq->q_points[i] - delta;
      s_codebook[2*i+1] = cosq->q_points[i] + delta;
    }
    temp = cosq->q_points;
    cosq->q_points = s_codebook;
    s_codebook = temp;
    s_levels <<= 1;
    rate++;
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, s_levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, s_levels*sizeof(double)));
    nnc_smem_size = 2 * s_levels * sizeof(double);
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_error_matrix(cosq->error_matrix, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->error_matrix, cosq->error_matrix, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    s_nnc_ge32<<<nnc_grid_size, nnc_block_size, nnc_smem_size>>>(s_levels, device->training_sequence,
        device->q_points, device->error_matrix, device->cc_cell_sums, device->cc_cardinality);
    cc_grid_size = {s_levels, 1, 1};
    cc_ge32<<<cc_grid_size, cc_block_size>>>(s_levels, device->q_points, device->error_matrix,
        device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cosq->q_points, device->q_points, sizeof(double) * s_levels, cudaMemcpyDeviceToHost));
  }
  free(s_codebook);
  free(cc_cell_sums);
  free(cc_cardinality);
}

/**
 * Allocate memory for device arrays.
 */
Device::Device(COSQ* cosq) {
  // Memory allocation
  checkCudaErrors(cudaMalloc((void **) &training_sequence, (cosq->training_size)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &error_matrix, (cosq->levels)*(cosq->levels)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &q_points, (cosq->levels)*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &q_cells, (cosq->training_size)*sizeof(unsigned int)));

  checkCudaErrors(cudaMalloc((void **) &cc_cardinality, (cosq->levels)*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &cc_cell_sums, (cosq->levels)*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &reduction_sums, (cosq->training_size)*sizeof(double)));

  // Memory copying
  checkCudaErrors(cudaMemcpy(training_sequence, cosq->training_sequence,
                            (cosq->training_size)*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(cc_cardinality, 0, (cosq->levels)*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(cc_cell_sums, 0, (cosq->levels)*sizeof(double)));

  // CUDA kernel launch params
  nnc_ge32_grid_size = {cosq->training_size, 1, 1};
  nnc_ge32_block_size = {WARP_SIZE, 1, 1};
  nnc_lt32_grid_size = {cosq->training_size * cosq->levels / WARP_SIZE, 1, 1};
  nnc_lt32_block_size = {WARP_SIZE, 1, 1};
  nnc_smem_size = 2 * cosq->levels * sizeof(double);

  cc_grid_size = {cosq->levels, 1, 1};
  cc_block_size = {WARP_SIZE, 1, 1};

  dist_grid_size = {cosq->training_size / WARP_SIZE, 1, 1};
  dist_block_size = {WARP_SIZE, 1, 1};
  dist_smem_size = cosq->levels * sizeof(double);
}

/**
 * Free all memory on device.
 */
Device::~Device() {
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
COSQ::COSQ(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate) {
  this->bit_rate = *bit_rate;
  this->levels = 1 << *bit_rate;
  this->training_sequence = training_sequence;
  this->training_size = *training_size;
  this->error_matrix = (double*) malloc(levels*levels*sizeof(double));
  if(COSQ::error_matrix == nullptr) {
    spdlog::error("Memory Allocation error: Failed to allocate memory for error_matrix!");
    return;
  }
  COSQ::q_points = (double*) malloc(levels*sizeof(double));
  if(COSQ::q_points == nullptr) {
    spdlog::error("Memory Allocation error: Failed to allocate memory for q_points!");
    return;
  }
  device = new Device(this);
}

COSQ::~COSQ() {
  free(error_matrix);
  free(q_points);
  delete device;
}

/**
 *
 */
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
 * TODO: Use CUDA to accelerate this
 */
void compute_error_matrix(double* error_matrix, unsigned int levels, unsigned int bit_rate) {
  for(int i = 0; i < levels; i++) {
    for(int j = 0; j < levels; j++) {
      error_matrix[j + i * levels] = polya_urn_error(j, i, bit_rate);
    }
  }
}

void cc_lt32(unsigned int levels, double* error_matrix, double* cc_sums,
    unsigned int* cc_cardinality, double* q_points) {
  double numerator = 0;
  double denominator = 0;
  for (int j = 0; j < levels; j++) {
    for (int i = 0; i < levels; i++) {
        numerator += error_matrix[j + levels * i] * cc_sums[i];
    }
    for (int i = 0; i < levels; i++) {
        denominator += error_matrix[j + levels * i] * cc_cardinality[i];
    }
    q_points[j] = numerator / denominator;
    numerator = 0;
    denominator = 0;
  }
}

/**
 *
 */
double* COSQ::cosq_lt32() {
  double dist_prev = DBL_MAX, dist_curr = 0;
  Split split(this, device);
  split.split_lt32();
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_error_matrix(error_matrix, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->error_matrix, error_matrix, levels * levels * sizeof(double), cudaMemcpyHostToDevice));
  // For sequential CC
  double* cc_sums_lt32 = (double*) malloc(sizeof(double) * levels);
  unsigned int* cc_cardinal_lt32 = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    nnc_lt32<<<device->nnc_lt32_grid_size, device->nnc_lt32_block_size>>>(levels, device->training_sequence, device->q_points,
        device->error_matrix, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cc_sums_lt32, device->cc_cell_sums, sizeof(double) * levels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cc_cardinal_lt32, device->cc_cardinality, sizeof(unsigned int) * levels, cudaMemcpyDeviceToHost));
    cc_lt32(levels, error_matrix, cc_sums_lt32, cc_cardinal_lt32, q_points);
    checkCudaErrors(cudaMemcpy(device->q_points, q_points, sizeof(double) * levels, cudaMemcpyHostToDevice));
    distortion_gather_lt32<<<device->dist_grid_size, device->dist_block_size>>>(levels, device->training_sequence,
        device->q_points, device->error_matrix, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    spdlog::info("Distortion is {:f}", dist_curr);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
  }
  // TODO: Return copy of COSQ::q_points
  return nullptr;
  free(cc_sums_lt32);
  free(cc_cardinal_lt32);
}

/**
 *
 */
double* COSQ::cosq_ge32() {
  double dist_prev = DBL_MAX, dist_curr = 0;
  Split split(this, device);
  split.split_ge32();
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_error_matrix(error_matrix, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->error_matrix, error_matrix, levels * levels * sizeof(double), cudaMemcpyHostToDevice));
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    nnc_ge32<<<device->nnc_ge32_grid_size, device->nnc_ge32_block_size, device->nnc_smem_size>>>(levels, device->training_sequence,
        device->q_points, device->error_matrix, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_ge32<<<device->cc_grid_size, device->cc_block_size>>>(levels, device->q_points, device->error_matrix,
        device->cc_cell_sums, device->cc_cardinality);
    distortion_gather_ge32<<<device->dist_grid_size, device->dist_block_size, device->dist_smem_size>>>(levels, device->training_sequence,
        device->q_points, device->error_matrix, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    spdlog::info("Distortion is {:f}", dist_curr);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
  }
  // TODO: Return copy of COSQ::q_points
  return nullptr;
}

/**
 *
 */
double* COSQ::train() {
  if(training_sequence == nullptr || training_size == 0) {
    spdlog::error("Failed to train COSQ: Invalid training sequence or size!");
    return nullptr;
  }
  if(levels >= 32) {
    return cosq_ge32();
  } else {
    return cosq_lt32();
  }
}
