#!/bin/bash
training_length=1048576

rm -rf build
cmake -S src/ -B build/ -DCMAKE_TOOLCHAIN_FILE=$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake
cd build
make
cp ../sequence_generator.cpp .
g++ sequence_generator.cpp -o sqg
./sqg $training_length
if [ $? -eq 0 ]; then
    for i in {1..10}
    do
        ./jaguar -b $i -t $training_length -f sequence > ../p$i
    done
fi
cd ..