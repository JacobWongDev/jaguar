#include <stdio.h>
#include <dirent.h>
#include <vector_types.h>
#include <math.h>
#include "training.cuh"
#include "../util/cuda_util.h"
#include "../util/pgm_util.h"
#include "../util/logger.h"
#include "../cuda/dct.cuh"
#include "../cuda/cosq.cuh"

void center_pixels(float* pixels, int stride, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            pixels[i*stride + j] -= 128.0f;
        }
    }
}

/**
 * @brief Create a COSQ based on provided training images
 *
 */
int train(const char* dir_name, const char* channel_name) {
    DIR* directory = NULL;
    struct dirent* entry = NULL;
    float* matrix = NULL;
    size_t pitch;
    int matrix_stride;
    dim3 cuda_grid_size;
    dim3 cuda_block_size;
    float* device_plane_src;
    float* device_plane_result;

    if((directory = opendir(dir_name)) == NULL) {
        logger_send("Could not open directory for training!", ERROR);
        return EXIT_FAILURE;
    }
    while((entry = readdir(directory)) != NULL) {
        if(entry->d_name[0] == '.')
            continue;
        char directory_file[500];
        snprintf(directory_file, sizeof(directory_file), "%s/%s", dir_name, entry->d_name);
        // Quantize image here
        pgm_image* image = load_image(directory_file);
        if(image == NULL)
            return EXIT_FAILURE;
        // Copy image matrix of unsigned chars to float matrix.
        matrix = MallocPlaneFloat(image->width, image->height, &matrix_stride);
        copy_plane(image->intensity, image->width, matrix, matrix_stride, image->width, image->height);
        // Want pixel values to have mean 0, so subtract 128 from each.
        center_pixels(matrix, matrix_stride, image->width, image->height);
        // Allocate memory on GPU for grayscale matrices
        checkCudaErrors(cudaMallocPitch((void **) &device_plane_src, &pitch, image->width * sizeof(float), image->height));
        checkCudaErrors(cudaMallocPitch((void **) &device_plane_result, &pitch, image->width * sizeof(float), image->height));
        pitch /= sizeof(float);

        // Copy image matrix to GPU
        checkCudaErrors(cudaMemcpy2D(
            device_plane_src, pitch * sizeof(float), matrix, matrix_stride * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyHostToDevice));
        /*
            DCT SETUP AND EXECUTION
        */
        cuda_grid_size = {
            image->width / KER2_BLOCK_WIDTH,
            image->height / KER2_BLOCK_HEIGHT,
            1
        };
        cuda_block_size = {
            8,
            KER2_BLOCK_WIDTH / 8,
            KER2_BLOCK_HEIGHT / 8
        };
        dct<<<cuda_grid_size, cuda_block_size>>>(device_plane_result, device_plane_src, (int) pitch);

        fprintf(stdout, "Matrix before DCT: \n");
        for(int i = 0; i < 8; i++) {
            fprintf(stdout, "[");
            for(int j = 8; j < 16; j++) {
                fprintf(stdout, "%f ", matrix[i*image->width + j]);
            }
            fprintf(stdout, "]\n");
        }

        // Copy resulting matrix back to host
        checkCudaErrors(cudaMemcpy2D(
            matrix, matrix_stride * sizeof(float), device_plane_result, pitch * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyDeviceToHost));

        fprintf(stdout, "Matrix after DCT: \n");
        for(int i = 0; i < 8; i++) {
            fprintf(stdout, "[");
            for(int j = 8; j < 16; j++) {
                fprintf(stdout, "%f ", matrix[i*image->width + j]);
            }
            fprintf(stdout, "]\n");
        }

        /*
            COSQ SETUP AND EXECUTION
        */
        // Swap pointers
        float* tmp = device_plane_result;
        device_plane_result = device_plane_src;
        device_plane_src = tmp;

        cuda_grid_size = {
            image->width / BLOCK_SIZE,
            image->height / BLOCK_SIZE,
            1
        };
        cuda_block_size = {1, 1, 1};
        int num_bits = 1;
        cosq<<<cuda_grid_size, cuda_block_size>>>(device_plane_result, device_plane_src, (int) pitch, num_bits, 1 << num_bits);;
        // Copy resulting matrix back to host
        checkCudaErrors(cudaMemcpy2D(
            matrix, matrix_stride * sizeof(float), device_plane_result, pitch * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyDeviceToHost));

        fprintf(stdout, "Matrix after COSQ: \n");
        for(int i = 0; i < 8; i++) {
            fprintf(stdout, "[");
            for(int j = 8; j < 16; j++) {
                fprintf(stdout, "%f ", matrix[i*image->width + j]);
            }
            fprintf(stdout, "]\n");
        }

        checkCudaErrors(cudaFree(device_plane_src));
        checkCudaErrors(cudaFree(device_plane_result));
        free_image(image);
    }
    return EXIT_SUCCESS;
}