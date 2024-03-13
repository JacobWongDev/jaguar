#!/bin/bash

make clean build
cd build
mkdir training-images
cp ../Lena.pgm training-images
make
./cosq -T -d training-images -c bs
cd ..