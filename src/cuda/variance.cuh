#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "common.h"


__global__ void variance(float* variance_matrix, int variance_matrix_pitch, float* matrix, int matrix_pitch, int image_width, int image_height) {
    float variance = 0;
    float average = 0;
    unsigned int sequence_length = (image_height * image_width / BLOCK_SIZE2);
    for(int i = blockIdx.y; i < image_height; i += BLOCK_SIZE) {
        for(int j = blockIdx.x; j < image_width; j += BLOCK_SIZE) {
            average += matrix[j + i*matrix_pitch];
        }
    }
    average /= sequence_length;
    for(int i = blockIdx.y; i < image_height; i += BLOCK_SIZE) {
        for(int j = blockIdx.x; j < image_width; j += BLOCK_SIZE) {
            variance += pow(matrix[j + i*matrix_pitch] - average, 2);
        }
    }
    variance_matrix[blockIdx.x + BLOCK_SIZE * blockIdx.y] = variance / sequence_length;
}