cmake_minimum_required(VERSION 3.22.1)
project(cosq LANGUAGES C CUDA)
set(CMAKE_C_STANDARD 23)

# Source Files
set(SOURCE_FILES
    jaguar.cu
    util/logger.c
    util/cuda_util.c
    util/pgm_util.c
    training/training.cu
)

# Header Files
include_directories(${CMAKE_SOURCE_DIR})

# External Dependencies
find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)