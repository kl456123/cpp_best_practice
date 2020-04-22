cmake_minimum_required(VERSION 3.2.0)

project(opencl)

find_package(OpenCL REQUIRED)

option(BUILD_C "BUILD C API" ON)
option(BUILD_CC "BUILD C++ API" ON)


if(BUILD_CC)
    add_executable(ocl_cc_demo opencl_demo.cc)
    target_link_libraries(ocl_cc_demo -lOpenCL)
endif()

if(BUILD_C)
    add_executable(ocl_c_demo opencl_demo.c)
    target_link_libraries(ocl_c_demo -lOpenCL)
endif()
