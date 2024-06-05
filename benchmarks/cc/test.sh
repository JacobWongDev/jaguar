#!/bin/bash

# make clean build
# cd build
# make
# ./cosq_benchmark
# cd ..

rm -rf cosq
g++ -fopenmp jaguar.cpp -o cosq -O3
./cosq