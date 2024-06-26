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
void COSQ::compute_error_matrix(double* error_matrix, unsigned int levels, unsigned int bit_rate) {
  for(int i = 0; i < levels; i++) {
    for(int j = 0; j < levels; j++) {
      error_matrix[j + i * levels] = polya_urn_error(j, i, bit_rate);
    }
  }
}

void COSQ::cc_lt32(double* cc_sums, unsigned int* cc_cardinality) {
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
  spdlog::info("NNC: Executing test on cells...");
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
  spdlog::info("NNC: Performing correctness test CC");
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
double* COSQ::cosq_lt32() {
  double dist_prev = DBL_MAX, dist_curr = 0;
  // For now, just use first few training seq elements
  for(int i = 0; i < levels; i++)
    q_points[i] = training_sequence[i];
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_error_matrix(error_matrix, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->error_matrix, error_matrix, levels * levels * sizeof(double), cudaMemcpyHostToDevice));

  // Testing data /////////////////////////////////////////////////////////////////////////////
  unsigned int* cpu_cells = (unsigned int*) malloc(sizeof(unsigned int) * training_size);
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * training_size);
  double* all_sums_nnc = (double*) malloc(sizeof(double) * training_size * levels);
  double* cpu_cc_cell_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cpu_cc_cardinal = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_cc_training_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cuda_cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_codebook = (double*) malloc(sizeof(double) * levels);
  memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
  memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);
  //////////////////////////////////////////////////////////////////////////////////////////////
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    // NNC
    nnc_lt32<<<device->nnc_lt32_grid_size, device->nnc_lt32_block_size>>>(levels, device->training_sequence, device->q_points,
        device->error_matrix, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    nnc_cpu(training_size, cpu_cells, training_sequence, q_points, levels, error_matrix, all_sums_nnc, cpu_cc_cell_sums, cpu_cc_cardinal);
    checkCudaErrors(cudaMemcpy(cuda_cells, device->q_cells, sizeof(unsigned int) * training_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, device->cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, device->cc_cell_sums, levels*sizeof(double), cudaMemcpyDeviceToHost));
    nnc_cells_test(training_size, levels, cuda_cells, cpu_cells, all_sums_nnc);
    nnc_cc_test(training_size, levels, training_sequence, cpu_cells, cuda_cc_cardinality, cuda_cc_training_sums);

    // CC
    cc_lt32(cpu_cc_cell_sums, cpu_cc_cardinal);
    checkCudaErrors(cudaMemcpy(device->q_points, q_points, sizeof(double) * levels, cudaMemcpyHostToDevice));
    spdlog::info("CC: Skipping test, using sequential impl!");

    // Distortion
    distortion_gather_lt32<<<device->dist_grid_size, device->dist_block_size>>>(levels, device->training_sequence,
        device->q_points, device->error_matrix, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    double d_cpu = distortion_cpu(training_size, levels, training_sequence, error_matrix, q_points, cpu_cells);
    if(fabsf64(d_cpu - dist_curr) > MAX_ERROR) {
      spdlog::error("Distortion test failed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    } else {
      spdlog::info("Distortion test passed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    }
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
    memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
    memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);
  }
  return nullptr;
  free(cpu_cells);
  free(cuda_cells);
  free(cuda_codebook);
  free(all_sums_nnc);
  free(cpu_cc_cell_sums);
  free(cpu_cc_cardinal);
  free(cuda_cc_training_sums);
  free(cuda_cc_cardinality);
}

double* COSQ::cosq_ge32() {
  double dist_prev = DBL_MAX, dist_curr = 0;
  // For now, just use first few training seq elements
  for(int i = 0; i < levels; i++)
    q_points[i] = training_sequence[i];
  checkCudaErrors(cudaMemcpy(device->q_points, q_points, levels * sizeof(double), cudaMemcpyHostToDevice));
  compute_error_matrix(error_matrix, levels, bit_rate);
  checkCudaErrors(cudaMemcpy(device->error_matrix, error_matrix, levels * levels * sizeof(double), cudaMemcpyHostToDevice));

  // Testing data /////////////////////////////////////////////////////////////////////////////
  unsigned int* cpu_cells = (unsigned int*) malloc(sizeof(unsigned int) * training_size);
  unsigned int* cuda_cells = (unsigned int*) malloc(sizeof(unsigned int) * training_size);
  double* all_sums_nnc = (double*) malloc(sizeof(double) * training_size * levels);
  double* cpu_cc_cell_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cpu_cc_cardinal = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_cc_training_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cuda_cc_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* cuda_codebook = (double*) malloc(sizeof(double) * levels);
  memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
  memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);
  //////////////////////////////////////////////////////////////////////////////////////////////
  // COSQ algorithm
  while(true) {
    checkCudaErrors(cudaMemset(device->cc_cardinality, 0, levels*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device->cc_cell_sums, 0, levels*sizeof(double)));
    // NNC
    nnc_ge32<<<device->nnc_ge32_grid_size, device->nnc_ge32_block_size, device->nnc_smem_size>>>(levels, device->training_sequence, device->q_points,
        device->error_matrix, device->q_cells, device->cc_cell_sums, device->cc_cardinality);
    nnc_cpu(training_size, cpu_cells, training_sequence, q_points, levels, error_matrix, all_sums_nnc, cpu_cc_cell_sums, cpu_cc_cardinal);
    checkCudaErrors(cudaMemcpy(cuda_cells, device->q_cells, training_size*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_cardinality, device->cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cuda_cc_training_sums, device->cc_cell_sums, levels*sizeof(double), cudaMemcpyDeviceToHost));
    nnc_cells_test(training_size, levels, cuda_cells, cpu_cells, all_sums_nnc);
    nnc_cc_test(training_size, levels, training_sequence, cpu_cells, cuda_cc_cardinality, cuda_cc_training_sums);
    // CC
    cc_ge32<<<device->cc_grid_size, device->cc_block_size>>>(levels, device->q_points, device->error_matrix,
        device->cc_cell_sums, device->cc_cardinality);
    cc_cpu(levels, error_matrix, cpu_cc_cell_sums, cpu_cc_cardinal, q_points);
    checkCudaErrors(cudaMemcpy(cuda_codebook, device->q_points, levels*sizeof(double), cudaMemcpyDeviceToHost));
    cc_correct(q_points, cuda_codebook, levels);
    // Distortion
    distortion_gather_ge32<<<device->dist_grid_size, device->dist_block_size, device->dist_smem_size>>>(levels, device->training_sequence,
        device->q_points, device->error_matrix, device->q_cells, device->reduction_sums);
    dist_curr = distortion_reduce(training_size, device->reduction_sums);
    double d_cpu = distortion_cpu(training_size, levels, training_sequence, error_matrix, q_points, cpu_cells);
    if(fabsf64(d_cpu - dist_curr) > MAX_ERROR) {
      spdlog::error("Distortion test failed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    } else {
      spdlog::info("Distortion test passed! CPU {:f} vs. GPU {:f}", d_cpu, dist_curr);
    }
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
    memset(cpu_cc_cell_sums, 0, sizeof(double) * levels);
    memset(cpu_cc_cardinal, 0, sizeof(unsigned int) * levels);
  }
  return nullptr;
  free(cpu_cells);
  free(cuda_cells);
  free(cuda_codebook);
  free(all_sums_nnc);
  free(cpu_cc_cell_sums);
  free(cpu_cc_cardinal);
  free(cuda_cc_training_sums);
  free(cuda_cc_cardinality);
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
