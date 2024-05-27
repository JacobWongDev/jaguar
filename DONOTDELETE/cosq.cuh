#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "common.h"

#define ERROR_PROBABILITY 0.1

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
  int x = a ^ b;
  int count = 0; // number of bits that differ
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return powf(ERROR_PROBABILITY, count) * powf(1-ERROR_PROBABILITY, num_bits - count);
  // if(a == b)
  //   return 1;
  // else
  //   return 0;
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
  cell* previous[MAX_CODEBOOK_SIZE];
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
  float partition_sizes[MAX_CODEBOOK_SIZE];
  float partition_sums[MAX_CODEBOOK_SIZE];
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
__device__ float distortion(int levels, int num_bits, cell** roots, float* codebook, int training_size) {
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
  return d / training_size;
}

namespace simple {
  __global__ void cosq(
      unsigned int * bit_allocations,
      float* normal_sequence,
      float* laplacian_sequence,
      float* device_quantizers,
      int quantizers_pitch,
      cell* device_regions,
      int regions_pitch) {
    int num_bits = bit_allocations[blockIdx.x];
    int levels = 1 << num_bits;
    float current_distortion = 0;
    float previous_distortion = 0;
    float codebook[MAX_CODEBOOK_SIZE]; // we will only use indexes up to int levels.
    cell* roots[MAX_CODEBOOK_SIZE];
    cell* regions = &device_regions[regions_pitch * blockIdx.x];
    float* quantizers = &device_quantizers[quantizers_pitch * blockIdx.x];
    nullify(roots, MAX_CODEBOOK_SIZE);
    float threshold = 0.01;
    if(blockIdx.x == 0) {
      // normal quantizer
      //setup regions
      for(int i = 0; i < TRAINING_SIZE; i++) {
        regions[i].value = &normal_sequence[i];
        regions[i].next = NULL;
      }
      // initialize codebook
      // Use first N training points as initial codebook.
      for(int i = 0; i < levels; i++) {
        codebook[i] = normal_sequence[i];
      }
    } else {
      // laplacian quantizer
      //setup regions
      for(int i = 0; i < TRAINING_SIZE; i++) {
        regions[i].value = &laplacian_sequence[i];
        regions[i].next = NULL;
      }
      // initialize codebook
      // Use first N training points as initial codebook.
      for(int i = 0; i < levels; i++) {
        codebook[i] = laplacian_sequence[i];
      }
    }

    // First iteration
    nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits);
    centroid(roots, codebook, levels, num_bits);
    previous_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE);
    // Lloyd Iteration
    while(1) {
      nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits);
      centroid(roots, codebook, levels, num_bits);
      current_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE);
      if((previous_distortion - current_distortion) / previous_distortion < threshold)
        break;
      previous_distortion = current_distortion;
    }

    // Write codebooks
    for(int i = 0; i < levels; i++) {
      quantizers[i] = codebook[i];
    }
    # if __CUDA_ARCH__ >= 200
      printf("I am thread %d,%d: bit allocation %d, Finished with distortion %f\n", blockIdx.x, blockIdx.y, num_bits, current_distortion);
    #endif
  }
}

// namespace complex {
//     __global__ void cosq(
//         float* dst,
//         float* src,
//         cell* device_regions,
//         unsigned int* bit_allocation,
//         int image_height,
//         int image_width,
//         int regions_pitch,
//         int bit_allocation_pitch,
//         int pitch) {
//         if(blockIdx.x != 0 || blockIdx.y != 0)
//             return;
//         int num_bits = 7;//bit_allocation[blockIdx.x + blockIdx.y * bit_allocation_pitch];
//         int levels = 1 << num_bits;
//         float current_distortion = 0;
//         float previous_distortion = 0;
//         int training_size = image_width * image_height / BLOCK_SIZE2;
//         float codebook[MAX_CODEBOOK_SIZE]; // we will only use indexes up to int levels.
//         cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
//         nullify(roots, MAX_CODEBOOK_SIZE);
//         //cell* regions = &device_regions[regions_pitch * (BLOCK_SIZE * blockIdx.y + blockIdx.x)]; // regions contain the actual values and pointers to next in list.
//         cell regions[1024];
//         float threshold = 0.01;
//         // setup proper memory access
//         src += blockIdx.y*pitch + blockIdx.x;
//         // dst += (image_height / BLOCK_SIZE)*blockIdx.y*pitch + (image_width / BLOCK_SIZE) * blockIdx.x;
//         //setup regions
//         bool e = false;
//         for(int h = 0; h < image_height / BLOCK_SIZE; h++) {
//             for(int w = 0; w < image_width / BLOCK_SIZE; w++) {
//             if(h * (image_width / BLOCK_SIZE) + w < training_size) {
//                 // # if __CUDA_ARCH__ >= 200
//                 //   printf("setting region %d to src %d\n", h * image_width / BLOCK_SIZE + w, h*BLOCK_SIZE*pitch + w*BLOCK_SIZE);
//                 // #endif
//                 regions[h * image_width / BLOCK_SIZE + w].value = &src[h*BLOCK_SIZE*pitch + w*BLOCK_SIZE];
//                 regions[h * image_width / BLOCK_SIZE + w].next = NULL;
//             } else {
//                 e = true;
//                 break;
//             }
//             }
//             if(e)
//             break;
//         }
//         // initialize codebook
//         // Use first N training points as initial codebook.
//         for(int h = 0; h < image_height / BLOCK_SIZE; h++) {
//             for(int w = 0; w < image_width / BLOCK_SIZE; w++) {
//             if(h * (image_width / BLOCK_SIZE) + w < levels) {
//                 // # if __CUDA_ARCH__ >= 200
//                 //   printf("setting codebook %d to src %d\n", h * image_width / BLOCK_SIZE + w, h*BLOCK_SIZE*pitch + w*BLOCK_SIZE);
//                 // #endif
//                 codebook[h * (image_width / BLOCK_SIZE) + w] = src[h*BLOCK_SIZE*pitch + w*BLOCK_SIZE];
//             } else {
//                 e = true;
//                 break;
//             }
//             }
//             if(e)
//             break;
//         }

//         // // First iteration
//         // nearest_neighbour(codebook, roots, levels, regions, training_size, num_bits);
//         centroid(roots, codebook, levels, num_bits);
//         previous_distortion = distortion(levels, num_bits, roots, codebook, training_size);
//         // nearest_neighbour(codebook, roots, levels, regions, training_size, num_bits, 2);
//         // # if __CUDA_ARCH__ >= 200
//         //   printf("Codebook!:\n");
//         //   for(int i = 0; i < levels; i++) {
//         //     printf("%f, ", codebook[i]);
//         //   }
//         //   printf("regions!:\n");
//         //   for(int i = 0; i < 1024; i++) {
//         //     printf("%f, ", *(regions[i].value));
//         //   }
//         // #endif
//         nearest_neighbour(codebook, roots, levels, regions, training_size, num_bits);
//         //Lloyd Iteration
//         // while(1) {
//         //   nearest_neighbour(codebook, roots, levels, regions, training_size, num_bits);
//         //   centroid(roots, codebook, levels, num_bits);
//         //   current_distortion = distortion(levels, num_bits, roots, codebook, training_size);
//         //   if((previous_distortion - current_distortion) / previous_distortion < threshold)
//         //     break;
//         //   previous_distortion = current_distortion;
//         //   // # if __CUDA_ARCH__ >= 200
//         //   //   printf("p_dist %f, c_dist %f", previous_distortion, current_distortion);
//         //   // #endif
//         // }
//         // // save codebook to dst
//         // int k = 0;
//         // for(int h = 0; h < image_height / BLOCK_SIZE; h++) {
//         //   for(int w = 0; w < image_width / BLOCK_SIZE; w++) {
//         //     if(h * (image_width / BLOCK_SIZE) + w < levels) {
//         //       dst[k++] = codebook[h * (image_width / BLOCK_SIZE) + w];
//         //     } else {
//         //       return;
//         //     }
//         //   }
//         //   k += pitch - (image_width / BLOCK_SIZE);
//         // }
//     }
// }