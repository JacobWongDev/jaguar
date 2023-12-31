#include <stdio.h>
#include <dirent.h>
#include <vector_types.h>
#include "training.cuh"
#include "../util/cuda_util.h"
#include "../util/pgm_util.h"
#include "../util/logger.h"
#include "../cuda/dct.cuh"

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
    size_t device_stride;
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
        checkCudaErrors(cudaMallocPitch((void **) &device_plane_src, &device_stride, image->width * sizeof(float), image->height));
        checkCudaErrors(cudaMallocPitch((void **) &device_plane_result, &device_stride, image->width * sizeof(float), image->height));
        device_stride /= sizeof(float);
        checkCudaErrors(cudaMemcpy2D(
            device_plane_src, device_stride * sizeof(float), matrix, matrix_stride * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyHostToDevice));
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
        dct<<<cuda_grid_size, cuda_block_size>>>(device_plane_result, device_plane_src, (int) device_stride);

        fprintf(stdout, "Matrix before DCT: \n");
        for(int i = 0; i < 8; i++) {
            fprintf(stdout, "[");
            for(int j = 8; j < 16; j++) {
                fprintf(stdout, "%f ", matrix[i*image->width + j]);
            }
            fprintf(stdout, "]\n");
        }

        checkCudaErrors(cudaMemcpy2D(
            matrix, matrix_stride * sizeof(float), device_plane_result, device_stride * sizeof(float),
            image->width * sizeof(float), image->height, cudaMemcpyDeviceToHost));

        fprintf(stdout, "Matrix after DCT: \n");
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