cmake_minimum_required(VERSION 3.22.1)
project(cosq LANGUAGES C CUDA)
set(CMAKE_C_STANDARD 23)

# Source Files
set(SOURCE_FILES
    jaguar.c
    logger.c
    cuda/cuda_util.c
)

# Header Files
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cuda_util
)

# External Dependencies
find_package(CUDAToolkit REQUIRED)
# find_package(spdlog CONFIG REQUIRED)
# find_package(PNG REQUIRED)
# find_package(Boost 1.83 REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
# target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)
# target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})