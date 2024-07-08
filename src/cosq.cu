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

void Split::split_lt5() {
  double* temp = NULL;
  double* s_codebook = (double*) malloc(sizeof(double) * cosq->levels);
  cosq->q_points = (double*) malloc(sizeof(double) * cosq->levels);
  // Compute centroid of training sequence
  double sum = 0;
  for(int i = 0; i < cosq->training_size; i++)
      sum += cosq->training_sequence[i];
  cosq->q_points[0] = sum / cosq->training_size;
  unsigned int rate = 0;
  unsigned int s_levels = 1;
  nnc_lt5_block_size = {1024, 1, 1};
  cc_gather_block_size = {64, 1, 1};
  cc_gather_smem_size = (cc_gather_block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));
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
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_ctm(cosq->ctm, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->ctm, cosq->ctm, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    nnc_lt5_grid_size = {cosq->training_size / (nnc_lt5_block_size.x / s_levels), 1, 1};
    nnc_lt5<<<nnc_lt5_grid_size, nnc_lt5_block_size>>>
        (s_levels, device->training_sequence, device->q_points, device->ctm, device->q_cells);
    cc_gather_grid_size = {s_levels, 1, 1};
    cc_gather<<<cc_gather_grid_size, cc_gather_block_size, cc_gather_smem_size>>>
        (device->training_sequence, cosq->training_size, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_le5_grid_size = {s_levels, 1, 1};
    cc_le5_block_size = {s_levels, 1, 1};
    cc_le5<<<cc_le5_grid_size, cc_le5_block_size>>>
        (s_levels, device->q_points, device->ctm, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cosq->q_points, device->q_points, s_levels * sizeof(double), cudaMemcpyDeviceToHost));
  }
  free(s_codebook);
}

void Split::split_ge5() {
  double* temp = NULL;
  double* s_codebook = (double*) malloc(sizeof(double) * cosq->levels);
  cosq->q_points = (double*) malloc(sizeof(double) * cosq->levels);
  // Compute centroid of training sequence
  double sum = 0;
  for(int i = 0; i < cosq->training_size; i++)
      sum += cosq->training_sequence[i];
  cosq->q_points[0] = sum / cosq->training_size;
  unsigned int rate = 0;
  unsigned int s_levels = 1;
  nnc_lt5_block_size = {1024, 1, 1};
  cc_gather_block_size = {64, 1, 1};
  cc_gather_smem_size = (cc_gather_block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));
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
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_ctm(cosq->ctm, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->ctm, cosq->ctm, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    nnc_lt5_grid_size = {cosq->training_size / (nnc_lt5_block_size.x / s_levels), 1, 1};
    nnc_lt5<<<nnc_lt5_grid_size, nnc_lt5_block_size>>>
        (s_levels, device->training_sequence, device->q_points, device->ctm, device->q_cells);
    cc_gather_grid_size = {s_levels, 1, 1};
    cc_gather<<<cc_gather_grid_size, cc_gather_block_size, cc_gather_smem_size>>>
        (device->training_sequence, cosq->training_size, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_le5_grid_size = {s_levels, 1, 1};
    cc_le5_block_size = {s_levels, 1, 1};
    cc_le5<<<cc_le5_grid_size, cc_le5_block_size>>>
        (s_levels, device->q_points, device->ctm, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cosq->q_points, device->q_points, s_levels * sizeof(double), cudaMemcpyDeviceToHost));
  }
  nnc_ge5_block_size = {1024, 1, 1};
  nnc_ge5_smem_size = (nnc_ge5_block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));
  cc_ge5_block_size = {32, 1, 1};
  cc_ge5_smem_size = 2 * (cc_ge5_block_size.x / WARP_SIZE) * sizeof(double);
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
    checkCudaErrors(cudaMemcpy(device->q_points, cosq->q_points, sizeof(double) * s_levels, cudaMemcpyHostToDevice));
    compute_ctm(cosq->ctm, s_levels, rate);
    checkCudaErrors(cudaMemcpy(device->ctm, cosq->ctm, sizeof(double) * s_levels * s_levels, cudaMemcpyHostToDevice));
    nnc_ge5_grid_size = {cosq->training_size / (nnc_ge5_block_size.x / s_levels), 1, 1};
    nnc_ge5<<<nnc_ge5_grid_size, nnc_ge5_block_size, nnc_ge5_smem_size>>>
        (s_levels, device->training_sequence, device->ctm, device->q_points, device->q_cells);
    cc_gather_grid_size = {s_levels, 1, 1};
    cc_gather<<<cc_gather_grid_size, cc_gather_block_size, cc_gather_smem_size>>>
        (device->training_sequence, cosq->training_size, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_ge5_grid_size = {s_levels, 1, 1};
    cc_ge5<<<cc_ge5_grid_size, cc_ge5_block_size, cc_ge5_smem_size>>>
        (s_levels, device->q_points, device->ctm, device->cc_cell_sums, device->cc_cardinality);
    checkCudaErrors(cudaMemcpy(cosq->q_points, device->q_points, s_levels * sizeof(double), cudaMemcpyDeviceToHost));
  }
  free(s_codebook);
}

/**
 * Allocate memory for device arrays.
 */
Device::Device(COSQ* cosq) {
  // Memory allocation
  checkCudaErrors(cudaMalloc((void **) &training_sequence, (cosq->training_size)*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **) &ctm, (cosq->levels)*(cosq->levels)*sizeof(double)));
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
  nnc_ge5_block_size = {1024, 1, 1};
  nnc_ge5_grid_size = {cosq->training_size / (nnc_ge5_block_size.x / cosq->levels), 1, 1};
  nnc_ge5_smem_size = (nnc_ge5_block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));

  nnc_lt5_block_size = {1024, 1, 1};
  nnc_lt5_grid_size = {cosq->training_size / (nnc_lt5_block_size.x / cosq->levels), 1, 1};

  cc_gather_grid_size = {cosq->levels, 1, 1};
  cc_gather_block_size = {64, 1, 1};
  cc_gather_smem_size = (cc_gather_block_size.x / WARP_SIZE) * (sizeof(double) + sizeof(unsigned int));

  cc_ge5_grid_size = {cosq->levels, 1, 1};
  cc_ge5_block_size = {32, 1, 1};
  cc_ge5_smem_size = 2 * (cc_ge5_block_size.x / WARP_SIZE) * sizeof(double);

  cc_le5_grid_size = {cosq->levels, 1, 1};
  cc_le5_block_size = {cosq->levels, 1, 1};

  dist_block_size = {1024, 1, 1};
  dist_grid_size = {cosq->training_size / dist_block_size.x, 1, 1};
  dist_smem_size = sizeof(double) * cosq->levels;
}

/**
 * Free all memory on device.
 */
Device::~Device() {
  checkCudaErrors(cudaFree(training_sequence));
  checkCudaErrors(cudaFree(ctm));
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
  this->ctm = (double*) malloc(levels*levels*sizeof(double));
  if(COSQ::ctm == nullptr) {
    spdlog::error("Memory Allocation error: Failed to allocate memory for ctm!");
    return;
  }
  device = new Device(this);
}

COSQ::~COSQ() {
  free(ctm);
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
void compute_ctm(double* ctm, unsigned int levels, unsigned int rate) {
  for(int i = 0; i < levels; i++) {
    for(int j = 0; j < levels; j++) {
      ctm[i + j * levels] = polya_urn_error(j, i, rate);
    }
  }
}

/**
 *
 */
void COSQ::cosq_lt5(double* target_q_points) {
  double dist_prev = DBL_MAX, dist_curr = 0;
  Split split(this, device);
  split.split_lt5();
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_ctm(ctm, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->ctm, ctm, levels * levels * sizeof(double), cudaMemcpyHostToDevice));
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    nnc_lt5<<<device->nnc_lt5_grid_size, device->nnc_lt5_block_size>>>
        (levels, device->training_sequence, device->q_points, device->ctm, device->q_cells);
    cc_gather<<<device->cc_gather_grid_size, device->cc_gather_block_size, device->cc_gather_smem_size>>>
        (device->training_sequence, training_size, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_le5<<<device->cc_le5_grid_size, device->cc_le5_block_size>>>
        (levels, device->q_points, device->ctm, device->cc_cell_sums, device->cc_cardinality);
    distortion_gather<<<device->dist_grid_size, device->dist_block_size, device->dist_smem_size>>>
        (levels, device->training_sequence, device->q_points, device->ctm, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    spdlog::info("Distortion is {:f}", dist_curr);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
  }
  memcpy(target_q_points, q_points, sizeof(double) * levels);
}

/**
 *
 */
void COSQ::cosq_ge5(double* target_q_points) {
  double dist_prev = DBL_MAX, dist_curr = 0;
  Split split(this, device);
  split.split_ge5();
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_ctm(ctm, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->ctm, ctm, levels * levels * sizeof(double), cudaMemcpyHostToDevice));
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    nnc_ge5<<<device->nnc_ge5_grid_size, device->nnc_ge5_block_size, device->nnc_ge5_smem_size>>>
        (levels, device->training_sequence, device->ctm, device->q_points, device->q_cells);
    cc_gather<<<device->cc_gather_grid_size, device->cc_gather_block_size, device->cc_gather_smem_size>>>
        (device->training_sequence, training_size, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    cc_ge5<<<device->cc_ge5_grid_size, device->cc_ge5_block_size, device->cc_ge5_smem_size>>>
        (levels, device->q_points, device->ctm, device->cc_cell_sums, device->cc_cardinality);
    distortion_gather<<<device->dist_grid_size, device->dist_block_size, device->dist_smem_size>>>
        (levels, device->training_sequence, device->q_points, device->ctm, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    spdlog::info("Distortion is {:f}", dist_curr);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
  }
  checkCudaErrors(cudaMemcpy(q_points, device->q_points, levels * sizeof(double), cudaMemcpyDeviceToHost));
  memcpy(target_q_points, q_points, sizeof(double) * levels);
}

/**
 *
 */
void COSQ::train(double* target_q_points) {
  if(training_sequence == nullptr || training_size == 0) {
    spdlog::error("Failed to train COSQ: Invalid training sequence or size!");
  }
  if(bit_rate < 5) {
    cosq_lt5(target_q_points);
  } else {
    cosq_ge5(target_q_points);
  }
}
