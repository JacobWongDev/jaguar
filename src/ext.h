#include "util/cuda_util.h"
#include "cuda/nvidia.cuh"
#include "spdlog/spdlog.h"

#define MIN(a, b) ((a < b) ? a : b)
unsigned int nextPow2(unsigned int x);
bool isPow2(unsigned int x);
void getNumBlocksAndThreads(int n, int maxBlocks,
        int maxThreads, int &blocks, int &threads);
void reduce(int size, int threads, int blocks, double *device_seq, double* device_res);
double distortion_reduce(unsigned int training_size, double* device_reduce_sums);