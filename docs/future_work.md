# Future Work & Improvements

This document describes work that can be pursued to improve this project.

## Code quality improvements

1. Determine the cause of long build time.

  Currently the CMake configure step takes about a minute. I am not sure whether this is due to the
  fact that I am using vcpkg as a dependency manager and that adds latency, or something else, but the
  build time takes too long.

2. Is FP64 the correct choice?

  All data types in Jaguar are doubles. Originally, I wrote the code to use floats, however,
  when I tested the accuracy of Jaguar, the final codebook was always different from the sequential
  implementation. Doing further testing, I compared the results of the sequential execution to Jaguar
  after each COSQ iteration (NNC, CC, Distortion, repeat), and the results at each step deviated very early on.

  I realized this could be due to float point summation inaccuracy, so even after using the Kahan summation in Jaguar
  and the sequential version, there was still a discrepancy between the results.

  So I gave up on using float and switched to double, and my tests passed. One could investigate
  the use of mixed precision calculations instead of just using pure doubles.

3. Jaguar can crash

  Jaguar will crash when certain distributions are provided. The reason is due to empty quantization cells.
  After the NNC, each training element should have an associated codebook element. The problem occurs when a codebook
  element does not have any training elements associated with it (its quantization cell is empty).

  In the centroid condition, we divide by the number of elements in the quantization cell and this will raise an error
  since we are dividing by zero.

  The splitting technique is used to try and avoid such situations, but I imagine that one can easily crash Jaguar
  if the provided distribution isn't sparse enough.

  For instance, passing a training sequence consisting of the same number 2^20 times.

## Performance improvements

Below are some ideas on what can be done to improve Jaguar's performance.

### New CUDA Kernels

1. Write CUDA kernel to compute channel transition matrix.
2. Write CUDA kernel to implement splitting of codebook in splitting technique.
   This will reduce the number of cudaMemcpy calls.

### Different Hardware (Assuming Jaguar stays with FP64)

The RTX 2070 has a FP32 to FP64 performance ratio of 1:32. In other words,
to do a FP64 operation it takes 32x longer than a FP32 operation. Using a GPU
with better FP64 performance can yield better results.

### Shared memory and Constant memory

In a few kernels, including the distortion_gather and nnc*, I tried storing the codebook (q_points)
in shared memory but didn't notice any major preformance benefits. For now I have omitted the use
of shared memory, but I think it can provide improvements if more profiling is done to determine
why there isnt a difference. I suspect the FP64 operations are a bottleneck on performance.

I also tried to do the same with constant memory, by putting the channel transition matrix (ctm)
in constant memory when the bit rate was <= 6 (64KB is the max constant memory size, so bit rate 6
is the largest ctm that can be accomodated). Doing this also did not present significant benefits,
so the code was discareded.

One has to be careful when using constant memory however, since any space used for constant mem reduces cache space.
Since I didn't observe any significant benefits when storing the ctm in constant memory, I opted to leave
all 64KB for caching purposes.

### Granularity of Parallelism

The level of parallelism needs to be explored further. For the NNC CUDA Kernels I wrote a few different ones
since the NNC was the operation that took the longest in the sequential impl and hence required the most
attention to improve performance. However, for other kernels like distortion_gather, I wrote it such that each
thread only computes 1 sum, no more and no less. This resulted in good speedup, but I'm sure if more time was
invested there are other kernel configurations that could work better

Ex. Each WARP handles 1 sum, not just 1 thread.
Ex. Each Block handles a sum, not just 1 thread.
