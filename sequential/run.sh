#!/bin/bash
rm -f cosq
g++ cosq.cpp -o cosq -g
 ./cosq
# valgrind -s --track-origins=yes --leak-check=yes ./cosq
