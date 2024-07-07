#!/bin/bash

make clean build
cd build
make
./cosq_benchmark
# ncu --set full --target-processes all --launch-skip 0 --launch-count 1 ./cosq_benchmark > ../report.ncu
cd ..