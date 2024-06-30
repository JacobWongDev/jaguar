#!/bin/bash
training_length=1048576
bit_rate=3
rm -f cosq
g++ cosq.cpp -o cosq -g
 ./cosq $bit_rate $training_length
# valgrind -s --track-origins=yes --leak-check=yes ./cosq