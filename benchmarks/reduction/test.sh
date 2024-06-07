#!/bin/bash

make clean build
cd build
make
./reduction
cd ..