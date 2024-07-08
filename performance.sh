#!/bin/bash
training_length=1048576
bit_rate=6

rm -rf build
cmake -S src/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
cp ../sequence_generator.cpp .
g++ sequence_generator.cpp -o sqg
./sqg $training_length
if [ $? -eq 0 ]; then
    ./jaguar -b $bit_rate -t $training_length -f sequence
fi
cd ..