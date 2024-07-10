
NNC Parallelism
- Break down N sums of length N. Each sum of length N can be split across different
threads and then reduced to produce a final result

Already done:
- Each thread gets one sum to calculate
- Each block handles more than one codebook element

- (Shared memory on NNC) For my RTX 2070, I noticed storing the q_points array in shared memory did not provide any significant performance benefits
so I did not include that. (I only tested for NNC rate 5)
- (Distortion) distortion code parallelism can be broken down further. As a simplifying assumption I decided to make each thread do 1 complete sum.
- (Wasted Cache) Should separate kernels such that the tm array in constant memory does not waste cache when it isnt used
- (constant memory) I observed that there wasnt much of an improvement when using constant mem on nnc OR distortion but where was for cc. I only tested a few bit rates,
but I opted to not use constant memory since
    - It is reserved for the entire application lifetime, and the amount of constant mem must be known at compile time. So each application execution will run on reduced cache even if its not the correct bitrate to leverage it
    - It only benefits the CC (2x speedup).
- (Hardware limitations) The performance ratio of FP32 to FP64 operations is 1:32 for the RTX 2070.

- WILL break if cells are empty

- Look into why CMake configure step takes almost a minute

- Better GPU for double precision calculations