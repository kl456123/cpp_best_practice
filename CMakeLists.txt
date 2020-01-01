cmake_minimum_required(VERSION 2.8)

project(DLCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


option(BUILD_TEST "build tests or not" ON)
option(BUILD_SHARED_LIBS "build shared library" ON)


find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)




set(SRC_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/src)
set(SRC_DIR
    ${SRC_ROOT_PATH}/core
    ${SRC_ROOT_PATH}/backends/cpu
    ${SRC_ROOT_PATH}/backends/opencl
    ${SRC_ROOT_PATH}/ops
)

set(INCLUDE_DIR ${SRC_DIR} ${SRC_ROOT_PATH})


message(STATUS "${INCLUDE_DIR}")
include_directories(${INCLUDE_DIR})


# *.c
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_c ${file_path}/*.c ${file_path}/*.h)
    set(SRC_FILES ${SRC_FILES} ${file_source_c})
endforeach()
message(STATUS "[*] searching *.c")
 
# *.cc
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_cc ${file_path}/*.cc ${file_path}/*.hpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cc})
endforeach()
message(STATUS "[*] searching *.cc")
 
# *.cpp
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_cpp ${file_path}/*.cpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cpp})
endforeach()
message(STATUS "[*] searching *.cpp")


if(BUILD_SHARED_LIBS)
    add_library(DLCL ${SRC_FILES})
    target_link_libraries(DLCL ${OpenCV_LIBS} -lOpenCL -pthread)
endif()



if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

