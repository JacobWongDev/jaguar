#!/bin/bash
rm -f cosq
g++ cosq.cpp -o cosq -g
# ./cosq 0 0 4
valgrind -s --track-origins=yes --leak-check=yes ./cosq 0 0 4