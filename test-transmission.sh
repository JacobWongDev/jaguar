#!/bin/bash

make clean build
cd build
mkdir imgs
cp ../Lena.pgm imgs
cp ../default.quantizer .
make
./cosq -C -d imgs -q default.quantizer -c bsc
convert Lena_test.pgm test.png
cd ..