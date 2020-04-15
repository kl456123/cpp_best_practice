cmake_minimum_required(VERSION 3.2.0)

project(cuda VERSION 1.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)


find_package(CUDA QUIET REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

message(${CUDA_LIBRARIES})


add_executable(demo_main ${CMAKE_CURRENT_SOURCE_DIR}/cuda/examples/demo_main.cc)
target_link_libraries(demo_main ${CUDA_LIBRARIES})
