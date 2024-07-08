#include <cuda_device_runtime_api.h>

/*
  Each block handles <blockDim.x> number of sums.
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