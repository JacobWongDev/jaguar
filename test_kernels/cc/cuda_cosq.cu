#include <random>
#include <chrono>
#include <iostream>
#include "cosq.cuh"

#define TRAINING_SIZE 1048576
#define POLYA_EPSILON 0.01
#define POLYA_DELTA 0
#define MAX_ERROR 0.0000001
#define ITER 11

void check(cudaError_t error, const char* file, int line) {
    if(cudaSuccess != error) {
        printf("CUDA error in %s: line %d code=%d(%s): %s\n", file, line, (unsigned int) error, cudaGetErrorName(error), cudaGetErrorString(error));
    }
}

#define checkCudaErrors(error) check(error, __FILE__, __LINE__);

void cc_cpu(int levels, double* error_matrix, double* cc_sums, unsigned int* cc_cardinality, double* codebook) {
    double numerator = 0;
    double denominator = 0;
    for (int j = 0; j < levels; j++) {
        for (int i = 0; i < levels; i++) {
            numerator += error_matrix[i + levels * j] * cc_sums[i];
            denominator += error_matrix[i + levels * j] * cc_cardinality[i];
        }
        codebook[j] = numerator / denominator;
        numerator = 0;
        denominator = 0;
    }
}

inline double polya_urn_error(int j, int i, int num_bits) {
    double temp;
    int x = j ^ i;
    int previous;
    if (x & 1 == 1) {
        temp = POLYA_EPSILON;
        previous = 1;
    } else {
        temp = 1 - POLYA_EPSILON;
        previous = 0;
    }
    x >>= 1;
    for (int i = 1; i < num_bits; i++) {
        if (x & 1 == 1) {
            temp *= (POLYA_EPSILON + previous * POLYA_DELTA) / (1 + POLYA_DELTA);
            previous = 1;
        } else {
            temp *= ((1 - POLYA_EPSILON) + (1 - previous) * POLYA_DELTA) / (1 + POLYA_DELTA);
            previous = 0;
        }
        x >>= 1;
    }
    return temp;
}

/**
 * Computes channel transition matrix p(j|i) where
 * i is the input symbol
 * j is the output symbol
 *
 * To promote coalesced memory access on the GPU, the matrix
 * is calculated in transposed form
 *
 * Typical: p(j|i) = mat[j + n*i]
 *
 * Transposed access: p(j|i) = mat[i + n*j]
 *
 */
double* compute_error_matrix(unsigned int levels, unsigned int rate) {
  double* error_matrix = (double*) malloc(sizeof(double) * levels * levels);
  for(int i = 0; i < levels; i++) {
      for(int j = 0; j < levels; j++) {
          error_matrix[i + j * levels] = polya_urn_error(j, i, rate);
      }
  }
  return error_matrix;
}

void cc_correct(double* codebook_seq, double* codebook_cuda, unsigned int levels) {
    bool correct = true;
    for (int i = 0; i < levels; i++) {
        if (abs(codebook_seq[i] - codebook_cuda[i]) > MAX_ERROR) {
            printf("The codebooks DO NOT match!\n");
            printf("Disagreement at %d: codebook_seq %f, codebook gpu %f", i, codebook_seq[i], codebook_cuda[i]);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("The codebooks match! CC Correctness test passed!\n");
}

int main(int argc, char **argv) {
    unsigned int rate = atoi(argv[1]);
    const unsigned int levels = 1 << rate;
    double *error_matrix = compute_error_matrix(levels, rate);
    double *codebook_seq = (double *)malloc(sizeof(double) * levels);
    double *codebook_cuda = (double *)malloc(sizeof(double) * levels);
    double *cc_training_sums = (double *)calloc(levels, sizeof(double));
    unsigned int *cc_cardinality = (unsigned int *)calloc(levels, sizeof(unsigned int));
    double* device_cc_training_sums;
    unsigned int* device_cc_cardinality;
    checkCudaErrors(cudaMalloc((void **) &device_cc_training_sums, levels*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &device_cc_cardinality, levels*sizeof(double)));
    double* dummy_cc_training_sums = (double *)malloc(sizeof(double) * levels);
    unsigned int* dummy_cc_cardinality = (unsigned int *)malloc(sizeof(unsigned int) * levels);
    // dummy variables to add memcpy latency
    double* dummy_device_codebook;
    checkCudaErrors(cudaMalloc((void **) &dummy_device_codebook, levels*sizeof(double)));
    // intialize codebook to first <levels> training samples
    // initialize training_sums and cc_cardinality
    std::default_random_engine rng;
    std::uniform_int_distribution<int> distribution(0, levels - 1);
    std::uniform_real_distribution<double> distribution_d(1, 100);
    rng.seed(31);
    for (int i = 0; i < levels; i++) {
        cc_training_sums[i] = distribution_d(rng);
        cc_cardinality[i] = (double)distribution(rng);
    }
    /*****************************************************************************************
     * Tests for Centroid Condition
     *****************************************************************************************/
    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::nanoseconds exec_time;
    int sum = 0;
    /*
      Sequential CC
    */
    std::cout << ":::::::::::: Performance CPU-only code ::::::::::::" << std::endl;
    for(int i = 0; i < ITER; i++) {
        start = std::chrono::high_resolution_clock::now();
        // Simulate copying of data from GPU to CPU (in practice this is necessary if we want to use sequential impl)
        checkCudaErrors(cudaMemcpy(dummy_cc_training_sums, device_cc_training_sums, levels*sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(dummy_cc_cardinality, device_cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cc_cpu(levels, error_matrix, cc_training_sums, cc_cardinality, codebook_seq);
        checkCudaErrors(cudaMemcpy(dummy_device_codebook, codebook_seq, levels*sizeof(double), cudaMemcpyHostToDevice));
        end = std::chrono::high_resolution_clock::now();
        exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        if(i == 0) {
            std::cout << "Warm-up time is " << exec_time.count() << "ns." << std::endl;
        } else {
            sum += exec_time.count();
        }
    }
    std::cout << "The average of the remaining exec times is " << sum / (ITER - 1) << "ns." << std::endl;

    /*
        Cuda CC
    */
    double* device_codebook;
    double* device_error_matrix;
    checkCudaErrors(cudaMalloc((void **) &device_codebook, levels*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &device_error_matrix, levels*levels*sizeof(double)));
    checkCudaErrors(cudaMemcpy(device_error_matrix, error_matrix, levels*levels*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_cc_training_sums, cc_training_sums, levels*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_cc_cardinality, cc_cardinality, levels*sizeof(unsigned int), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpyToSymbol(tm, error_matrix, levels*levels*sizeof(double)));
    sum = 0;
    std::cout << ":::::::::::: Performance GPU-only code ::::::::::::" << std::endl;
    for(int i = 0; i < ITER; i++) {
        if(rate <= 5) {
            start = std::chrono::high_resolution_clock::now();
            dim3 block_size = {levels, 1, 1};
            dim3 grid_size = {levels, 1, 1};
            // cc_le5<<<grid_size, block_size>>>(levels, device_codebook, device_cc_training_sums, device_cc_cardinality);
            cc_le5<<<grid_size, block_size>>>(levels, device_codebook, device_error_matrix, device_cc_training_sums, device_cc_cardinality);
            checkCudaErrors(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            if(i == 0) {
                std::cout << "Warm-up time is " << exec_time.count() << "ns." << std::endl;
            } else {
                sum += exec_time.count();
            }
        } else {
            start = std::chrono::high_resolution_clock::now();
            dim3 block_size = {32, 1, 1};
            dim3 grid_size = {levels, 1, 1};
            unsigned int smem_size = 2 * (block_size.x / WARP_SIZE) * sizeof(double);
            cc_gt5<<<grid_size, block_size, smem_size>>>(levels, device_codebook, device_error_matrix, device_cc_training_sums, device_cc_cardinality);
            checkCudaErrors(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            if(i == 0) {
                std::cout << "Warm-up time is " << exec_time.count() << "ns." << std::endl;
            } else {
                sum += exec_time.count();
            }
        }
    }
    std::cout << "The average of the remaining exec times is " << sum / (ITER - 1) << "ns." << std::endl;
    checkCudaErrors(cudaMemcpy(codebook_cuda, device_codebook, levels*sizeof(double), cudaMemcpyDeviceToHost));
    printf(":::::::::::: Performing correctness test CC ::::::::::::\n");
    // for(int i = 0; i < levels; i++)
    //     printf("%f ", codebook_cuda[i]);
    // printf("\n");
    // for(int i = 0; i < levels; i++)
    //     printf("%f ", codebook_seq[i]);
    cc_correct(codebook_seq, codebook_cuda, levels);
    checkCudaErrors(cudaFree(dummy_device_codebook));
    checkCudaErrors(cudaFree(device_codebook));
    checkCudaErrors(cudaFree(device_error_matrix));
    checkCudaErrors(cudaFree(device_cc_training_sums));
    checkCudaErrors(cudaFree(device_cc_cardinality));
    free(dummy_cc_training_sums);
    free(dummy_cc_cardinality);
    free(codebook_seq);
    free(codebook_cuda);
    free(cc_cardinality);
    free(cc_training_sums);
    free(error_matrix);
}