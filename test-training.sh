#!/bin/bash

make clean build
cd build
mkdir training-images
cp ../test_image.pgm training-images
make
./cosq -T -d training-images -c bs
cd ..