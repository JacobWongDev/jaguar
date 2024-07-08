#!/bin/bash
rm -rf build
cmake -S src/test/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
if [ $? -eq 0 ]; then
    for bit_rate in {1..10}
    do
        ./jaguar $bit_rate > ../br$bit_rate.txt
    done
fi
cd ..