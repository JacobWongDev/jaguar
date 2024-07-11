#include <cuda_device_runtime_api.h>

/**
 * @brief 1st half of distortion calculation
 *
 * Each thread calculates summation of length <levels>. This is
 * then reduced by distortion_reduce to a single value.
 *
 * Sums are stored in <intermediate>
 *
 * Kernel Requirements
 * blockDim.x is a power of 2.
 */
__global__ void distortion_gather(unsigned int levels, double* training_sequence, double* q_points,
    double* ctm, unsigned int* cells, double* intermediate) {
  extern __shared__ double s_q_points[];
  unsigned int t = threadIdx.x;
  unsigned int t_ = t + blockIdx.x * blockDim.x;
  unsigned int r = levels / blockDim.x;
  double target = training_sequence[t_];
  unsigned int i_nnc = cells[t_];
  double sum = 0;
  if(r == 0) {
    if(t < levels) {
      s_q_points[t] = q_points[t];
    }
  } else {
    for(int i = 0; i < r; i++) {
      s_q_points[t + blockDim.x * i] = q_points[t + blockDim.x * i];
    }
  }
  __syncthreads();
  //Perform summation
  for(int j = 0; j < levels; j++) {
    sum += ctm[i_nnc + levels*j] * (target - s_q_points[j]) * (target - s_q_points[j]);
  }
  intermediate[t_] = sum;
}