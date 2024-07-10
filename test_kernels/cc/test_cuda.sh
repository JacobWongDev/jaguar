#!/bin/bash

make clean build
cd build
make
for i in {1..10}
do
    ./cosq_benchmark $i > ../r$i
done
# ncu --set full --target-processes all --launch-skip 0 --launch-count 1 ./cosq_benchmark > ../report.ncu
cd ..