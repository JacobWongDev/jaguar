#!/bin/bash
bit_rate=6
rm -rf build
cmake -S src/test/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
if [ $? -eq 0 ]; then
    ./jaguar $bit_rate
fi
cd ..