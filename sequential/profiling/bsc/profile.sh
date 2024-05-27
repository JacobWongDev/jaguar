#!/bin/bash
num_normal=4
num_laplacian=4
epsilons=("0.0" "0.005" "0.01" "0.1")

rm -f cosq
g++ cosq.cpp -o cosq -pg
for epsilon in "${epsilons[@]}"; do
    echo "./cosq \"$epsilon\" \"$num_normal\" \"$num_laplacian\""
    ./cosq "$epsilon" $num_normal $num_laplacian
    gprof cosq > profile_${epsilon}
done