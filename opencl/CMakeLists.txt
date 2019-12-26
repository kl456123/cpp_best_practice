cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# add_executable(demo ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
include_directories(./OpenCLUtilities/)

add_library(opencl_backend ./OpenCLUtilities/context.cpp ./OpenCLUtilities/opencl_backend.cpp)

target_link_libraries(opencl_backend ${OpenCV_LIBS} -lOpenCL)
