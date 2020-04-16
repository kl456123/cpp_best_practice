cmake_minimum_required(VERSION 3.2.0)

project(cuda VERSION 1.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)


find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION OFF)
# set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# set(CUDA_HOST_COMPILER clang++)

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/examples/demo_main.cu)
# CUDA_COMPILE(CU_O ${CUDA_FILES})


cuda_add_executable(demo_main ${CUDA_FILES})
# add_executable(demo_main)
# target_link_libraries(${CU_O})
