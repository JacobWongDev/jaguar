/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "util/cuda_util.h"
#include "cuda/nvidia.cuh"
#include "spdlog/spdlog.h"

#define MIN(a, b) ((a < b) ? a : b)

unsigned int nextPow2(unsigned int x);

bool isPow2(unsigned int x);

/**
 * @brief Compute the number of threads and blocks to use for the given reduction
 * kernel. We set threads / block to the minimum of maxThreads and n/2. We observe
 * the maximum specified number of blocks, because each thread in that kernel can
 * process a variable number of elements.
 *
 * @param n
 * @param maxBlocks
 * @param maxThreads
 * @param blocks
 * @param threads
 */
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

/**
 * @brief Wrapper function for kernel launch
 *
 * @param size
 * @param threads
 * @param blocks
 * @param device_seq Input
 * @param device_res Result
 */
void reduce(int size, int threads, int blocks, double *device_seq, double* device_res);

/**
 * @brief Wrapper function for kernel launch
 *
 * @param training_size
 * @param device_reduce_sums intermediate sums from distortion_gather kernel
 */
double distortion_reduce(unsigned int training_size, double* device_reduce_sums);