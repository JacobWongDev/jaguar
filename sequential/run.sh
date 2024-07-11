#!/bin/bash
training_length=1048576
rm -f cosq
g++ cosq.cpp -o cosq -g
for bit_rate in {1..10}
do
    ./cosq $bit_rate $training_length > p$bit_rate
done
# valgrind -s --track-origins=yes --leak-check=yes ./cosq