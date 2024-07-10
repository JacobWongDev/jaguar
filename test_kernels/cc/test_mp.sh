#!/bin/bash

# make clean build
# cd build
# make
# ./cosq_benchmark
# cd ..

rm -rf cosq
g++ -fopenmp openmp_cosq.cpp -o cosq -O3
./cosq