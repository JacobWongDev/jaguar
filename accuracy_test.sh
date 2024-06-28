#!/bin/bash

rm -rf build
cmake -S src/test/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
if [ $? -eq 0 ]; then
    ./jaguar
fi
cd ..