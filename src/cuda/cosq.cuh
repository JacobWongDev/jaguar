#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "common.h"

struct quantizer_cell {
  struct quantizer_cell* next;
  float* value;
};

typedef struct quantizer_cell cell;

__device__ void nullify(cell** arr, int length) {
  for(int i = 0; i < length; i++)
    arr[i] = NULL;
}

/**
 * @brief Error measure used by the cosq.
 *
 * @param a
 * @param b
 * @return float
 */
__device__ float error(float a, float b) {
  return (a - b) * (a - b);
}

/**
 * @brief Channel error probability for the Binary Symmetric Channel.
 *
 */
__device__ float channel_error(int a, int b, int num_bits) {
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
__device__ void nearest_neighbour(float* codebook, cell** roots, int levels, cell* regions, int training_size, int num_bits) {
  float min = __FLT_MAX__;
  int min_index = -1;
  float sum = 0;
  cell* previous[BLOCK_SIZE2]; // so we don't have to traverse the linked list every time we want to insert a new element
  nullify(previous, levels);
  for(int i = 0; i < training_size; i++) {
    for(int l = 0; l < levels; l++) {
      for(int j = 0; j < levels; j++) {
        sum += channel_error(j, l, num_bits) * error(*(regions[i].value), codebook[j]);
      }
      if(sum < min) {
        min_index = l;
        min = sum;
      }
      sum=0;
    }
    // If first node in this partition:
    if(previous[min_index] == NULL) {
      roots[min_index] = &regions[i];
      previous[min_index] = roots[min_index];
    } else {
      (*previous[min_index]).next = &regions[i];
      previous[min_index] = &regions[i];
    }
    regions[i].next = NULL;
    //reset
    sum = 0;
    min_index = -1;
    min = __FLT_MAX__;
  }
}

/**
 * @brief Generalized centroid condition.
 *
 * @param regions
 * @param codebook
 * @return float
 */
__device__ void centroid(cell** roots, float* codebook, int levels, int num_bits) {
  float numerator = 0;
  float denominator = 0;
  float partition_sum = 0;
  float partition_sizes[BLOCK_SIZE2];
  float partition_sums[BLOCK_SIZE2];
  cell* ptr = NULL;
  int count = 0;

  // To save ourselves from calculating the same sum
  // for each codebook value, calculate it once and save the value.
  for(int i = 0; i < levels; i++) {
    ptr = roots[i];
    while(ptr != NULL) {
      partition_sum += *((*ptr).value);
      count++;
      ptr = ptr->next;
    }
    partition_sizes[i] = count;
    partition_sums[i] = partition_sum;
    count = 0;
    partition_sum = 0;
  }

  for(int i = 0; i < levels; i++) {
    // Compute Numerator
    for(int j = 0; j < levels; j++)
      numerator += channel_error(i, j, num_bits) * partition_sums[j];
    // Compute Denominator
    for(int j = 0; j < levels; j++)
      denominator += channel_error(i, j, num_bits) * partition_sizes[j];
    codebook[i] = numerator/denominator;
    numerator = 0;
    denominator = 0;
  }
}

/**
 * @brief
 *
 * @param levels
 * @param num_bits
 * @param training
 * @param codebook
 * @return float
 */
__device__ float distortion(int levels, int num_bits, cell** roots, float* codebook) {
  float d = 0;
  cell* traversal = NULL;
  for(int i = 0; i < levels; i++) {
    traversal = roots[i];
    while(traversal != NULL) {
      for(int j = 0; j < levels; j++) {
        d += channel_error(j, i, num_bits) * error(*(traversal->value), codebook[j]);
      }
      traversal = traversal->next;
    }
  }
  return d / BLOCK_SIZE2;
}


__global__ void cosq(float* dst, float* src, int pitch, int num_bits, int levels) {
  float current_distortion = 0;
  float previous_distortion = 0;
  float codebook[BLOCK_SIZE2];
  cell* roots[BLOCK_SIZE2]; // roots is used to point to the beginning of the linked list.
  nullify(roots, BLOCK_SIZE2);
  cell regions[BLOCK_SIZE2]; // regions contain the actual values and pointers to next in list.
  float threshold = 0.01;
  src += BLOCK_SIZE*blockIdx.y*pitch + BLOCK_SIZE*blockIdx.x;
  dst += BLOCK_SIZE*blockIdx.y*pitch + BLOCK_SIZE*blockIdx.x;
  //setup regions
  int k = 0;
  for(int i = 0; i < BLOCK_SIZE; i++) {
    for(int j = 0; j < BLOCK_SIZE; j++) {
      regions[i * BLOCK_SIZE + j].value = &src[k++];
      regions[i * BLOCK_SIZE + j].next = NULL;
    }
    k += pitch - BLOCK_SIZE;
  }
  // initialize codebook
  // Use first N training points as initial codebook.
  k = 0;
  bool e = false;
  for(int i = 0; i < BLOCK_SIZE; i++) {
    for(int j = 0; j < BLOCK_SIZE; j++) {
      if(i * BLOCK_SIZE + j < levels) {
        codebook[i * BLOCK_SIZE + j] = src[k++];
      } else {
        e = true;
        break;
      }
    }
    if(e)
      break;
    k += pitch - BLOCK_SIZE;
  }
  // First iteration
  nearest_neighbour(codebook, roots, levels, regions, BLOCK_SIZE2, num_bits);
  centroid(roots, codebook, levels, num_bits);
  previous_distortion = distortion(levels, num_bits, roots, codebook);
  // Lloyd Iteration
  while(1) {
    nearest_neighbour(codebook, roots, levels, regions, BLOCK_SIZE2, num_bits);
    centroid(roots, codebook, levels, num_bits);
    current_distortion = distortion(levels, num_bits, roots, codebook);
    if((previous_distortion - current_distortion) / previous_distortion < threshold)
      break;
    previous_distortion = current_distortion;
  }
  //save codebook to dst
  k = 0;
  for(int i = 0; i < BLOCK_SIZE; i++) {
    for(int j = 0; j < BLOCK_SIZE; j++) {
      if(i * BLOCK_SIZE + j < levels) {
        dst[k++] = codebook[i * BLOCK_SIZE + j];
      } else {
        return;
      }
    }
    k += pitch - BLOCK_SIZE;
  }
}