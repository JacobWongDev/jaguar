clean:
	rm -rf build

build:
	cmake -S src/ -B build/ -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_TOOLCHAIN_FILE=${PWD}/vcpkg/scripts/buildsystems/vcpkg.cmake

