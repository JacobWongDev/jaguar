#!/bin/bash
rm -rf profile_*
deltas=("0.0" "5.0" "10.0")
epsilons=("0.0" "0.005" "0.01" "0.1")
rate=4
rm -f cosq
g++ cosq.cpp -o cosq -pg
for delta in "${deltas[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        echo "./cosq \"$delta\" \"$epsilon\" \"$rate\""
        ./cosq "$delta" "$epsilon" "$rate"
        gprof cosq > profile_${delta}_${epsilon}
    done
done