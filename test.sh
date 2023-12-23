#!/bin/bash

make clean build
cp Lenna.pgm build
cd build
make
./cosq
cd ..