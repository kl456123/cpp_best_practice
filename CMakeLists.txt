cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_LIST_DIR}/pool)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/pool)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)



