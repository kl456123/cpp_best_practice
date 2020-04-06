cmake_minimum_required(VERSION 2.8)

project(DLCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


option(BUILD_TEST "build tests or not" ON)
option(BUILD_SHARED_LIBS "build shared library" ON)
option(BUILD_THIRD_PARTY "build third_party library" ON)


find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(GLOB ProtoFiles ${CMAKE_CURRENT_LIST_DIR}/src/core/protos/*.proto)
protobuf_generate_cpp(ProtoSources ProtoHeaders ${ProtoFiles})

# some third parties
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/abseil-cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/abseil-cpp)
set(ABSL_LIBRARIES absl::base absl::synchronization
    absl::strings absl::str_format_internal absl::container
    absl::hash absl::hashtablez_sampler Threads::Threads
    absl::flat_hash_map absl::debugging)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/nsync)
set(NSYNC_LIBRARIES -lnsync -lnsync_cpp)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/lib/common)
include_directories(${COMMON_INCLUDE_DIRS})
include_directories(${COMMON_PROTO_DIRS})


set(SRC_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/src)
set(SRC_DIR
    ${SRC_ROOT_PATH}/core
    ${SRC_ROOT_PATH}/backends/cpu
    ${SRC_ROOT_PATH}/backends/opencl
    ${SRC_ROOT_PATH}/ops
    ${SRC_ROOT_PATH}/core/platform
    ${SRC_ROOT_PATH}/core/protos # protos
)

set(INCLUDE_DIR ${SRC_ROOT_PATH})


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
    add_library(DLCL STATIC ${SRC_FILES} ${ProtoSources})
    target_link_libraries(DLCL ${OpenCV_LIBS} -lOpenCL -pthread ${Protobuf_LIBRARIES} common)
endif()




if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if(BUILD_THIRD_PARTY)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party)
endif()

