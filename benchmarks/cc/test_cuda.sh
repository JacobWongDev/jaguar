#!/bin/bash

make clean build
cd build
make
./cosq_benchmark
cd ..