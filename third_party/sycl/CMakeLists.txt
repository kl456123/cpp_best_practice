cmake_minimum_required(VERSION 3.10)

set(COMPUTECPP_DIR /usr/local/computecpp)
include_directories(${COMPUTECPP_DIR}/include)
link_directories(${COMPUTECPP_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++14")
# add_definitions(-DEIGEN_USE_SYCL)

# build eigen
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenGL REQUIRED)


add_executable(eigen eigen.cc)
add_executable(eigen_orig eigen_orig.cc)
# # add_executable(demo demo.cpp)

target_link_libraries(eigen -lComputeCpp -pthread -lGL)
target_link_libraries(eigen_orig -lComputeCpp -pthread)
