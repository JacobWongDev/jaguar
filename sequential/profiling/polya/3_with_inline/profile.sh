#!/bin/bash
num_normal=4
num_laplacian=4
deltas=("0.0" "5.0" "10.0")
epsilons=("0.0" "0.005" "0.01" "0.1")

rm -f cosq
g++ cosq.cpp -o cosq -pg
for delta in "${deltas[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        echo "./cosq \"$delta\" \"$epsilon\" \"$num_normal\" \"$num_laplacian\""
        ./cosq "$delta" "$epsilon" $num_normal $num_laplacian
        gprof cosq > profile_${delta}_${epsilon}
    done
done