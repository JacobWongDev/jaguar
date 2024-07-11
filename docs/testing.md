
# Testing

For this application, there are two kinds of tests:

1. Accuracy of results
2. Safety

## Accuracy Testing

Execute accuracy_test.sh

The accuracy of the program can be verified by comparing the result of the sequential implementation to the parallel version.
The COSQ algorithm is iterative, so at each step the data structures of both implementations are compared.

The testing code is in src/test. Essentially what I did is everytime I wanted to test Jaguar I copied the split*() and cosq*()
functions from src/cosq.cu src/cosq.h and pasted them in src/test/test_cosq.cu and src/test/test_cosq.h respectively.

Then I inserted calls to the sequential nnc, cc and distortion and called test methods to compare the data structures at runtime.

There are definitely better ways to test the code, but this was the easiest and most straight forward method for this small project.

If this project increases in size, a better testing methodology should be used.

## Safety Testing

Execute safe_test.sh

There are two types of memory that can have leaks:

1. GPU global memory
2. Host memory

To check GPU memory, the compute-sanitizer is used.

Valgrind could be used to verify the integrity of host memory, however
it has been documented that Valgrind will report CUDA memory allocations (and other operations)
as memory leaks. See [This link](https://stackoverflow.com/questions/20593450/valgrind-and-cuda-are-reported-leaks-real).

Instead, I simply counted the number of calls to malloc() and free() to make sure they matched in every file.
If the counts did not match, then there was likely a memory leak.
