#!/bin/bash

CUDA_TOOLS=("memcheck" "initcheck" "synccheck")

rm -rf build
cmake -S src/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
if [ $? -eq 0 ]; then
    for tool in "${CUDA_TOOLS[@]}"; do
        compute-sanitizer --tool=$tool --log-file=$tool.log ./jaguar
    done
fi
cd ..