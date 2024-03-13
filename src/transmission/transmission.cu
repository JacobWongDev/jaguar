#include <stdio.h>
#include <dirent.h>
#include "../util/cuda_util.hpp"
#include "../util/pgm_util.hpp"
#include "../util/logger.hpp"
#include "../cuda/dct.cuh"
#include "../cuda/bit_allocations.h"
#include "../cuda/variance.cuh"
#include "../cuda/quantize.cuh"

void center_pixels(float* pixels, int stride, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            pixels[i*stride + j] -= 128.0f;
        }
    }
}

void augment_pixels(float* pixels, int stride, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            pixels[i*stride + j] = floor(pixels[i*stride + j] + 128.0f);
        }
    }
}

float* read_quantizers(const char* quantizer_file, unsigned int** bit_allocation_map) {
    // Open the binary file for reading
    FILE *file = fopen(quantizer_file, "rb");
    unsigned int num_quantizers = 0;
    unsigned int quantizer_size = 0;
    if (file == NULL) {
        logger_send("Could not open quantizers file", log_level::ERROR);
    }
    // Read the number of quantizers from the file
    fread(&num_quantizers, sizeof(unsigned int), 1, file);
    float* quantizers = (float*) malloc(sizeof(float) * num_quantizers * MAX_CODEBOOK_SIZE);
    *bit_allocation_map = (unsigned int*) malloc(sizeof(unsigned int) * (num_quantizers + 1));
    (*bit_allocation_map)[0] = num_quantizers;
    float* temp = quantizers;
    if(quantizers == NULL) {
        logger_send("Could not allocate memory for quantizers", log_level::ERROR);
        return NULL;
    }
    // Read each quantizer from the file
    for (unsigned int i = 0; i < num_quantizers; i++) {
        // Read the quantizer size from the file in binary
        fread(&quantizer_size, sizeof(unsigned int), 1, file);
        (*bit_allocation_map)[i+1] = quantizer_size;
        // Allocate memory for the array
        // Read the array elements from the file in binary
        fread(temp, sizeof(float), quantizer_size, file);
        temp += MAX_CODEBOOK_SIZE;
    }
    // Close the file
    fclose(file);
    return quantizers;
}

void cuda_quantize(
        float* device_quantizers,
        int device_quantizers_pitch,
        unsigned int* device_bit_allocation_map,
        unsigned int* device_bit_allocation,
        float* device_variance_matrix,
        int device_variance_matrix_pitch,
        float* device_matrix,
        int matrix_pitch,
        unsigned int* device_quantized_matrix,
        int device_quantized_matrix_pitch,
        int image_width,
        int image_height) {
    dim3 cuda_grid_size;
    dim3 cuda_block_size;
    cuda_grid_size = {
        image_width / BLOCK_SIZE,
        image_height / BLOCK_SIZE,
        1
    };
    cuda_block_size = {BLOCK_SIZE, BLOCK_SIZE, 1};
    quantize<<<cuda_grid_size, cuda_block_size>>>(
        device_quantizers,
        device_quantizers_pitch,
        device_bit_allocation_map,
        device_bit_allocation,
        device_variance_matrix,
        device_variance_matrix_pitch,
        device_matrix,
        matrix_pitch,
        device_quantized_matrix,
        device_quantized_matrix_pitch,
        image_width,
        image_height);
}

void cuda_reverse_quantize(
        float* device_quantizers,
        int device_quantizers_pitch,
        unsigned int* device_bit_allocation_map,
        unsigned int* device_bit_allocation,
        float* device_variance_matrix,
        int device_variance_matrix_pitch,
        float* device_matrix,
        int matrix_pitch,
        unsigned int* device_quantized_matrix,
        int device_quantized_matrix_pitch,
        int image_width,
        int image_height) {
    dim3 cuda_grid_size;
    dim3 cuda_block_size;
    cuda_grid_size = {
        image_width / BLOCK_SIZE,
        image_height / BLOCK_SIZE,
        1
    };
    cuda_block_size = {BLOCK_SIZE, BLOCK_SIZE, 1};
    reverse_quantize<<<cuda_grid_size, cuda_block_size>>>(
        device_quantizers,
        device_quantizers_pitch,
        device_bit_allocation_map,
        device_bit_allocation,
        device_variance_matrix,
        device_variance_matrix_pitch,
        device_matrix,
        matrix_pitch,
        device_quantized_matrix,
        device_quantized_matrix_pitch,
        image_width,
        image_height);
}

// void cuda_channel_pass() {

// }

void cuda_variance(float* device_variance_matrix, int device_variance_matrix_pitch, float* device_matrix, int matrix_pitch, int image_width, int image_height) {
    dim3 cuda_grid_size;
    dim3 cuda_block_size;
    cuda_grid_size = {
        BLOCK_SIZE,
        BLOCK_SIZE,
        1
    };
    cuda_block_size = {1, 1, 1};
    variance<<<cuda_grid_size, cuda_block_size>>>(device_variance_matrix, device_variance_matrix_pitch, device_matrix, matrix_pitch, image_width, image_height);
}

void cuda_idct(
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
    idct<<<cuda_grid_size, cuda_block_size>>>(device_matrix_a, device_matrix_b, device_matrix_pitch);

    // Copy resulting matrix back to host
    checkCudaErrors(cudaMemcpy2D(
        local_matrix, local_matrix_pitch * sizeof(float), device_matrix_a, device_matrix_pitch * sizeof(float),
        image_width * sizeof(float), image_height, cudaMemcpyDeviceToHost));

    fprintf(stdout, "Matrix after IDCT: \n");
    for(int i = 0; i < 8; i++) {
        fprintf(stdout, "[");
        for(int j = 8; j < 16; j++) {
            fprintf(stdout, "%f ", local_matrix[i*image_width + j]);
        }
        fprintf(stdout, "]\n");
    }
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


int transmit_over_channel(unsigned int* bit_allocation, const char* dir_name, const char* quantizer_file, const char* channel_name) {
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
    float* device_variance_matrix;
    size_t device_variance_matrix_pitch;
    float* device_quantizers;
    unsigned int* device_quantized_matrix;
    size_t device_quantized_matrix_pitch;
    unsigned int* device_bit_allocations = NULL;
    unsigned int* device_bit_allocation_map = NULL;
    unsigned int* bit_allocation_map = NULL; // size of array is first index, then remaining are bit allocations
    // Read quantizers from file
    float* quantizers = read_quantizers(quantizer_file, &bit_allocation_map);
    // Setup bit allocation matrix
    checkCudaErrors(cudaMalloc((void **) &device_bit_allocations, BLOCK_SIZE2 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_bit_allocations, bit_allocation, BLOCK_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    // Copy bit allocation map to device
    checkCudaErrors(cudaMalloc((void **) &device_bit_allocation_map, (bit_allocation_map[0] + 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_bit_allocations, bit_allocation_map, (bit_allocation_map[0] + 1) * sizeof(float), cudaMemcpyHostToDevice));
    // Copy quantizers to device
    checkCudaErrors(cudaMalloc((void **) &device_quantizers, MAX_CODEBOOK_SIZE * bit_allocation_map[0] * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_quantizers, quantizers, bit_allocation_map[0] * MAX_CODEBOOK_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    if((directory = opendir(dir_name)) == NULL) {
        logger_send("Could not open directory for training!", log_level::ERROR);
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
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr,
                    "getLastCudaError() CUDA error :"
                    "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
        }
        /*
            Calculate variances of each pixel
        */
        checkCudaErrors(cudaMallocPitch((void **) &device_variance_matrix, &device_variance_matrix_pitch, BLOCK_SIZE * sizeof(float), BLOCK_SIZE));
        device_variance_matrix_pitch /= sizeof(float);
        cuda_variance(device_variance_matrix, device_variance_matrix_pitch, device_matrix_b, device_matrix_pitch, image->width, image->height);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr,
                    "getLastCudaError() CUDA error :"
                    "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
        }
        /*
            Quantize image
        */
        // Allocate memory for quantized matrix (unsigned ints)
        checkCudaErrors(cudaMallocPitch((void **) &device_quantized_matrix, &device_quantized_matrix_pitch, image->width * sizeof(unsigned int), image->height));
        device_quantized_matrix_pitch /= sizeof(unsigned int);
        cuda_quantize(
            device_quantizers,
            MAX_CODEBOOK_SIZE,
            device_bit_allocation_map,
            device_bit_allocations,
            device_variance_matrix,
            device_variance_matrix_pitch,
            device_matrix_b,
            device_matrix_pitch,
            device_quantized_matrix,
            device_quantized_matrix_pitch,
            image->width,
            image->height);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr,
                    "getLastCudaError() CUDA error :"
                    "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
        }
        /*
            Pass through channel
        */

        /*
            Reverse Quantize
        */
        cuda_reverse_quantize(
            device_quantizers,
            MAX_CODEBOOK_SIZE,
            device_bit_allocation_map,
            device_bit_allocations,
            device_variance_matrix,
            device_variance_matrix_pitch,
            device_matrix_b,
            device_matrix_pitch,
            device_quantized_matrix,
            device_quantized_matrix_pitch,
            image->width,
            image->height);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr,
                    "getLastCudaError() CUDA error :"
                    "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
        }
        /*
            IDCT
        */
        cuda_idct(local_matrix, device_matrix_b, device_matrix_a, image->width, image->height, (unsigned int) device_matrix_pitch, local_matrix_pitch);
        augment_pixels(local_matrix, local_matrix_pitch, image->width, image->height);
        save_image(image->height, image->width, local_matrix);
        checkCudaErrors(cudaFree(device_quantized_matrix));
        checkCudaErrors(cudaFree(device_bit_allocations));
        checkCudaErrors(cudaFree(device_matrix_a));
        checkCudaErrors(cudaFree(device_matrix_b));
        checkCudaErrors(cudaFree(device_variance_matrix));
        free_image(image);
    }
    free(quantizers);
    free(bit_allocation_map);
    return EXIT_SUCCESS;
}