cmake_minimum_required(VERSION 3.22.1)
project(cosq LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)

# Source Files
set(SOURCE_FILES
    jaguar.cu
    util/logger.cpp
    util/cuda_util.cpp
    util/pgm_util.cpp
    training/simple_training.cu
    transmission/transmission.cu
)

# Header Files
include_directories(${CMAKE_SOURCE_DIR})

# External Dependencies
find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)