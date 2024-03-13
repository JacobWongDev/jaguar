#include <unordered_set>
#include "../math/distributions.h"
#include "../util/logger.hpp"
#include "../cuda/common.h"
#include "../util/cuda_util.hpp"
#include "../cuda/cosq.cuh"

/**
 * Saves quantizer to a file.
*/
void save(float* quantizers, unsigned int* bit_allocations, unsigned int num_quantizers) {
    FILE *file = fopen("default.quantizer", "wb");
    if (file == NULL) {
        perror("Error opening file");
    } else {
        //Write number of arrays in file first
        fwrite(&num_quantizers, sizeof(unsigned int), 1, file);
        //Now write individual arrays to file
        for(int i = 0; i < num_quantizers; i++) {
            unsigned int k = 1 << bit_allocations[i];
            fwrite(&k, sizeof(unsigned int), 1, file);
            fwrite(quantizers, sizeof(float), k, file);
            quantizers += MAX_CODEBOOK_SIZE;
        }
    }
    fclose(file);
}

/**
 * @brief Create COSQs based on the following assumptions:
 *  - When Performing DCT on 8x8 pixel blocks:
 *      - AC coefficients are distributed according to the Laplacian PDF
 *      - DC coefficient is distributed according to the Normal PDF
 *
 */
int train(unsigned int* allocation_matrix) {
    // Host data
    unsigned int bit_allocations[BLOCK_SIZE2];
    unsigned int num_quantizers = 1;
    float* quantizers = NULL;
    // GPU pointers
    unsigned int* device_bit_allocations = NULL;
    float* device_normal_sequence = NULL;
    float* device_laplacian_sequence = NULL;
    float* device_quantizers = NULL;
    size_t device_quantizers_pitch;
    cell* device_regions = NULL;
    size_t device_regions_pitch;

    // First, determine how many quantizers need to be trained
    // Based on Bit Allocation Matrix

    // First entry is normally distributed
    bit_allocations[0] = allocation_matrix[0];

    std::unordered_set<unsigned int> seen;
    for(int i = 1; i < BLOCK_SIZE2; i++) {
        if(allocation_matrix[i] != 0 && !seen.contains(allocation_matrix[i])) {
            seen.insert(allocation_matrix[i]);
            bit_allocations[num_quantizers] = allocation_matrix[i];
            num_quantizers++;
        }
    }

    std::cout << "Bit allocations unique:" << std::endl;
    for(int i = 0; i < num_quantizers; i++)
        std::cout << bit_allocations[i] << std::endl;
    // Generate training sequences
    // TODO: skip this step in the future and just use integral calculations in centroid.
    float* normal_sequence = generate_normal_sequence();
    float* laplacian_sequence = generate_laplacian_sequence();


    // Verify 0 mean and unit variance::
    float sum = 0;
    float n_avg = 0;
    float n_var = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += normal_sequence[i];
    n_avg = sum / TRAINING_SIZE;

    std::cout << "Normal sequence E[X] = " << n_avg << std::endl;

    // Variance
    sum = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += pow(normal_sequence[i] - n_avg, 2);
    n_var = sum/TRAINING_SIZE;

    std::cout << "Normal sequence Var(X) = " << n_var << std::endl;

    sum = 0;
    float l_avg = 0;
    float l_var = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += laplacian_sequence[i];
    l_avg = sum / TRAINING_SIZE;

    std::cout << "Laplacian sequence E[X] = " << l_avg << std::endl;
    // Variance
    sum = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += pow(laplacian_sequence[i] - l_avg, 2);
    l_var = sum/TRAINING_SIZE;

    std::cout << "Laplacian sequence Var(X) = " << l_var << std::endl;

    // Copy bit allocations to GPU
    checkCudaErrors(cudaMalloc((void **) &device_bit_allocations, num_quantizers * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_bit_allocations, bit_allocations, num_quantizers * sizeof(float), cudaMemcpyHostToDevice));
    // Copy training sequences to GPU
    checkCudaErrors(cudaMalloc((void **) &device_normal_sequence, TRAINING_SIZE * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_normal_sequence, normal_sequence, TRAINING_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &device_laplacian_sequence, TRAINING_SIZE * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_laplacian_sequence, laplacian_sequence, TRAINING_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    // Allocate memory for quantizer (arrays of quantization points) returned from training
    checkCudaErrors(cudaMallocPitch((void **) &device_quantizers, &device_quantizers_pitch, MAX_CODEBOOK_SIZE * sizeof(float), num_quantizers));
    device_quantizers_pitch /= sizeof(float);
    // Allocate memory for local quantizer array
    quantizers = (float*) malloc(MAX_CODEBOOK_SIZE * num_quantizers * sizeof(float));

    /*
        "Stack" memory

        Allocate memory for algorithm since it won't fit on the CUDA stack
    */
    checkCudaErrors(cudaMallocPitch((void **) &device_regions, &device_regions_pitch, TRAINING_SIZE * sizeof(cell), num_quantizers));
    device_regions_pitch /= sizeof(cell);


    // Execute work on GPU
    dim3 cuda_grid_size = {num_quantizers, 1, 1};
    dim3 cuda_block_size = {1, 1, 1};

    simple::cosq<<<cuda_grid_size, cuda_block_size>>>(
        device_bit_allocations,
        device_normal_sequence,
        device_laplacian_sequence,
        device_quantizers,
        (int) device_quantizers_pitch,
        device_regions,
        (int) device_regions_pitch
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,
                "getLastCudaError() CUDA error :"
                "(%d) %s.\n",static_cast<int>(err),cudaGetErrorString(err));
    }

    // Copy quantizers back
    checkCudaErrors(cudaMemcpy2D(
        quantizers,
        MAX_CODEBOOK_SIZE * sizeof(float),
        device_quantizers,
        device_quantizers_pitch * sizeof(float),
        MAX_CODEBOOK_SIZE * sizeof(float),
        num_quantizers,
        cudaMemcpyDeviceToHost
    ));

    for(int i = 0; i < num_quantizers; i++) {
        int k = bit_allocations[i];
        printf("Printing array for bit allocation %d\n[", k);
        for(int j = 0; j < (1 << k); j++) {
            printf("%f,", quantizers[i * MAX_CODEBOOK_SIZE + j]);
        }
        printf("]\n");
    }

    save(quantizers, bit_allocations, num_quantizers);

    checkCudaErrors(cudaFree(device_quantizers));
    checkCudaErrors(cudaFree(device_regions));
    checkCudaErrors(cudaFree(device_bit_allocations));
    checkCudaErrors(cudaFree(device_laplacian_sequence));
    checkCudaErrors(cudaFree(device_normal_sequence));
    free(quantizers);
    free(normal_sequence);
    free(laplacian_sequence);
    return EXIT_SUCCESS;
}