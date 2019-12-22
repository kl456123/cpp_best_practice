cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


add_executable(demo ${CMAKE_CURRENT_LIST_DIR}/main.cpp)


message(${OpenCV_INCLUDE_DIRS})

target_link_libraries(demo ${OpenCV_LIBS} -lOpenCL)
