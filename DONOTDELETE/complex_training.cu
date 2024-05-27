#pragma once
#include <stdio.h>
#include <dirent.h>
#include <vector_types.h>
#include <math.h>
#include <time.h>
#include "complex_training.cuh"
#include "../util/cuda_util.hpp"
#include "../util/pgm_util.hpp"
#include "../util/logger.hpp"
#include "../cuda/dct.cuh"
#include "../cuda/cosq.h"
#include "../cuda/bit_allocations.h"

void center_pixels(float* pixels, int stride, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            pixels[i*stride + j] -= 128.0f;
        }
    }
}

void cuda_cosq(
        float* local_matrix,
        float* device_matrix_b,
        float* device_matrix_a,
        unsigned int image_width,
        unsigned int image_height,
        unsigned int device_matrix_pitch,
        unsigned int local_matrix_pitch) {
    dim3 cuda_grid_size = {BLOCK_SIZE, BLOCK_SIZE, 1};
    dim3 cuda_block_size = {1, 1, 1};
    cell* device_plane_regions = NULL;
    size_t regions_pitch;
    unsigned int* bit_allocation = NULL;
    size_t bit_allocation_pitch;
    // checkCudaErrors(cudaMallocPitch((void **) &device_plane_regions, &regions_pitch, BLOCK_SIZE2 * sizeof(cell), (MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT)/BLOCK_SIZE2));
    // checkCudaErrors(cudaMallocPitch((void **) &bit_allocation, &bit_allocation_pitch, BLOCK_SIZE * sizeof(unsigned int), BLOCK_SIZE));
    // checkCudaErrors(cudaMemcpy2D(
    //         bit_allocation, bit_allocation_pitch, __test, BLOCK_SIZE * sizeof(unsigned int),
    //         BLOCK_SIZE * sizeof(unsigned int), BLOCK_SIZE, cudaMemcpyHostToDevice));
    // Convert pitch from bytes to number of elements
    // regions_pitch /= sizeof(cell);
    // bit_allocation_pitch /= sizeof(unsigned int);

    // size_t free, total;
    // cudaError_t err = cudaMemGetInfo(&free, &total);
    // if (cudaSuccess != err) {
    // fprintf(stderr,
    //         "getLastCudaError() CUDA error :"
    //         "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
    // }
    // printf("Free: %f, Total %f", (double) free / (1024 * 1024), (double) total / (1024 * 1024));

    // Copy resulting matrix back to host
    complex::cosq<<<cuda_grid_size, cuda_block_size>>>(
        device_matrix_b,
        device_matrix_a,
        device_plane_regions,
        bit_allocation,
        image_height,
        image_width,
        (int) regions_pitch,
        (int) bit_allocation_pitch,
        (int) device_matrix_pitch
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,
                "getLastCudaError() CUDA error :"
                "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
    }
    // Copy resulting matrix back to host
    // checkCudaErrors(cudaMemcpy2D(
    //     local_matrix, local_matrix_pitch * sizeof(float), device_matrix_b, device_matrix_pitch * sizeof(float),
    //     image_width * sizeof(float), image_height, cudaMemcpyDeviceToHost));

    // fprintf(stdout, "Matrix after COSQ: \n");
    // for(int i = 0; i < image_height; i++) {
    //     fprintf(stdout, "[");
    //     for(int j = 0; j < image_width; j++) {
    //         fprintf(stdout, "%f ", local_matrix[i * image_width + j]);
    //     }
    //     fprintf(stdout, "]\n");
    // }
    // checkCudaErrors(cudaFree(bit_allocation));
    // checkCudaErrors(cudaFree(device_plane_regions));
}

void cuda_dct(
        float* local_matrix,
        float* device_matrix_b,
        float* device_matrix_a,
        unsigned int image_width,
        unsigned int image_height,
        unsigned int device_matrix_pitch,
        unsigned int local_matrix_pitch) {
    dim3 cuda_grid_size;
    dim3 cuda_block_size;
    cuda_grid_size = {
        image_width / KER2_BLOCK_WIDTH,
        image_height / KER2_BLOCK_HEIGHT,
        1
    };
    cuda_block_size = {
        8,
        KER2_BLOCK_WIDTH / 8,
        KER2_BLOCK_HEIGHT / 8
    };
    dct<<<cuda_grid_size, cuda_block_size>>>(device_matrix_b, device_matrix_a, device_matrix_pitch);

    fprintf(stdout, "Matrix before DCT: \n");
    for(int i = 0; i < 8; i++) {
        fprintf(stdout, "[");
        for(int j = 8; j < 16; j++) {
            fprintf(stdout, "%f ", local_matrix[i*image_width + j]);
        }
        fprintf(stdout, "]\n");
    }

    // Copy resulting matrix back to host
    checkCudaErrors(cudaMemcpy2D(
        local_matrix, local_matrix_pitch * sizeof(float), device_matrix_b, device_matrix_pitch * sizeof(float),
        image_width * sizeof(float), image_height, cudaMemcpyDeviceToHost));

    fprintf(stdout, "Matrix after DCT: \n");
    for(int i = 0; i < 8; i++) {
        fprintf(stdout, "[");
        for(int j = 8; j < 16; j++) {
            fprintf(stdout, "%f ", local_matrix[i*image_width + j]);
        }
        fprintf(stdout, "]\n");
    }
}

void save_cosq(float* local_matrix, unsigned int image_height, unsigned int image_width) {
    time_t t;
    time(&t);
    FILE *file = fopen(ctime(&t), "w");
    if(file == NULL) {
        logger_send("Could not save COSQs to file", ERROR);
    }
    for(int h = 0; h < image_height / BLOCK_SIZE; h++) {
        for(int w = 0; w < image_width / BLOCK_SIZE; w++) {

        }
    }
    fclose(file);
}

/**
 * @brief Create COSQs based on provided training images
 *
 */
int train(const char* dir_name, const char* channel_name) {
    /*
        Variables for Host
    */
    DIR* directory = NULL;
    struct dirent* entry = NULL;
    float* local_matrix = NULL;
    unsigned int local_matrix_pitch;
    /*
        Variables for GPU
    */
    float* device_matrix_a;
    float* device_matrix_b;
    size_t device_matrix_pitch;

    if((directory = opendir(dir_name)) == NULL) {
        logger_send("Could not open directory for training!", ERROR);
        return EXIT_FAILURE;
    }
    while((entry = readdir(directory)) != NULL) {
        /*
            Check for images in folder
        */
        if(entry->d_name[0] == '.')
            continue;
        char directory_file[500];
        snprintf(directory_file, sizeof(directory_file), "%s/%s", dir_name, entry->d_name);
        // Read image
        pgm_image* image = load_image(directory_file);
        if(image == NULL)
            return EXIT_FAILURE;
        /*
            Copy greyscale matrix to gpu memory
        */
        local_matrix = MallocPlaneFloat(image->width, image->height, &local_matrix_pitch);
        copy_plane(image->intensity, image->width, local_matrix, local_matrix_pitch, image->width, image->height);
        // Want pixel values to have mean 0, so subtract 128 from each.
        center_pixels(local_matrix, local_matrix_pitch, image->width, image->height);
        // Allocate memory on GPU for grayscale matrices
        checkCudaErrors(cudaMallocPitch((void **) &device_matrix_a, &device_matrix_pitch, image->width * sizeof(float), image->height));
        checkCudaErrors(cudaMallocPitch((void **) &device_matrix_b, &device_matrix_pitch, image->width * sizeof(float), image->height));
        device_matrix_pitch /= sizeof(float);
        // Copy image matrix to GPU
        checkCudaErrors(cudaMemcpy2D(
            device_matrix_a, device_matrix_pitch * sizeof(float), local_matrix, local_matrix_pitch * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyHostToDevice));
        /*
            DCT SETUP AND EXECUTION
        */
        cuda_dct(local_matrix, device_matrix_b, device_matrix_a, image->width, image->height, (unsigned int) device_matrix_pitch, local_matrix_pitch);
        /*
            COSQ SETUP AND EXECUTION
        */
        // Swap pointers
        float* tmp = device_matrix_b;
        device_matrix_b = device_matrix_a;
        device_matrix_a = tmp;
        cuda_cosq(local_matrix, device_matrix_b, device_matrix_a, image->width, image->height, (unsigned int) device_matrix_pitch, local_matrix_pitch);
        /*
            Free allocated GPU memory
        */
        free_image(image);
    }
    /*
     * Write trained COSQs to text file
    */

    checkCudaErrors(cudaFree(device_matrix_a));
    checkCudaErrors(cudaFree(device_matrix_b));
    return EXIT_SUCCESS;
}