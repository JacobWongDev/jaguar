#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include <stdio.h>

__device__ int search(int bits, unsigned int* bit_map) {
    for(int i = 1; i < bit_map[0] + 1; i++) {
        if(bit_map[i] == bits) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Error measure used by the cosq.
 *
 * @param a
 * @param b
 * @return float
 */
__device__ float error_(float a, float b) {
  return (a - b) * (a - b);
}

/**
 * @brief Channel error probability for the Binary Symmetric Channel.
 *
 */
__device__ float channel_error_(int a, int b, int num_bits) {
  // int x = a ^ b;
  // int count = 0; // number of bits that differ
  // float p = 0.2; // probability of error
  // while (x) {
  //   count += x & 1;
  //   x >>= 1;
  // }
  // return pow(p, count) * pow(1-p, num_bits - count);
  if(a == b)
    return 1;
  else
    return 0;
}

/**
 * @brief Generalized nearest neighbour condition.
 *
 * @param codebook
 * @param regions
 * @return float
 */
__device__ unsigned int nn(float* codebook, int levels, int num_bits, float value) {
  float min = __FLT_MAX__;
  int min_index = -1;
  float sum = 0;
  for(int l = 0; l < levels; l++) {
    for(int j = 0; j < levels; j++) {
      sum += channel_error_(j, l, num_bits) * error_(value, codebook[j]);
    }
    if(sum < min) {
      min_index = l;
      min = sum;
    }
    sum=0;
    min_index = -1;
    min = __FLT_MAX__;
  }
  return min_index;
}

__global__ void reverse_quantize(
        float* device_quantizers,
        int device_quantizers_pitch,
        unsigned int* device_bit_allocation_map,
        unsigned int* bit_allocation,
        float* variance_matrix,
        int variance_matrix_pitch,
        float* matrix,
        int matrix_pitch,
        unsigned int* quantized_matrix,
        int quantized_matrix_pitch,
        int image_width,
        int image_height) {
    int bits = bit_allocation[threadIdx.x + threadIdx.y * BLOCK_SIZE];

    int block = blockIdx.x * BLOCK_SIZE + blockIdx.y * BLOCK_SIZE * matrix_pitch;
    int matrix_index = block + threadIdx.x + threadIdx.y * matrix_pitch;

    int block_quantized_matrix = blockIdx.x * BLOCK_SIZE + blockIdx.y * BLOCK_SIZE * quantized_matrix_pitch;
    int quantized_matrix_index = block_quantized_matrix + threadIdx.x + threadIdx.y * quantized_matrix_pitch;
    int quantizer_index = search(bits, device_bit_allocation_map);

    float* quantizer = &device_quantizers[device_quantizers_pitch * quantizer_index];
    if(bits == 0) {
        quantized_matrix[quantized_matrix_index] = 0;
        return;
    }
    float variance = variance_matrix[threadIdx.x + threadIdx.y * BLOCK_SIZE];
    // Use NN to quantize value
    matrix[matrix_pitch] = variance * quantizer[quantized_matrix[quantized_matrix_index]];
}

__global__ void quantize(
        float* device_quantizers,
        int device_quantizers_pitch,
        unsigned int* device_bit_allocation_map,
        unsigned int* bit_allocation,
        float* variance_matrix,
        int variance_matrix_pitch,
        float* matrix,
        int matrix_pitch,
        unsigned int* quantized_matrix,
        int quantized_matrix_pitch,
        int image_width,
        int image_height) {
    int bits = bit_allocation[threadIdx.x + threadIdx.y * BLOCK_SIZE];

    int block = blockIdx.x * BLOCK_SIZE + blockIdx.y * BLOCK_SIZE * matrix_pitch;
    int matrix_index = block + threadIdx.x + threadIdx.y * matrix_pitch;

    int block_quantized_matrix = blockIdx.x * BLOCK_SIZE + blockIdx.y * BLOCK_SIZE * quantized_matrix_pitch;
    int quantized_matrix_index = block_quantized_matrix + threadIdx.x + threadIdx.y * quantized_matrix_pitch;
    int quantizer_index = search(bits, device_bit_allocation_map);

    float* quantizer = &device_quantizers[device_quantizers_pitch * quantizer_index];
    if(bits == 0) {
        quantized_matrix[quantized_matrix_index] = 0;
        return;
    }
    float variance = variance_matrix[threadIdx.x + threadIdx.y * BLOCK_SIZE];
    // // Use NN to quantize value
    quantized_matrix[quantized_matrix_index] = nn(quantizer, (1 << bits), bits, matrix[matrix_index] / variance);
    // # if __CUDA_ARCH__ >= 200
    //   printf("Made it here! %d %d", blockIdx.x, blockIdx.y);
    // #endif
}