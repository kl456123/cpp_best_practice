cmake_minimum_required(VERSION 2.8)

project(memory)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")




find_package(OpenCL REQUIRED)
find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(GLOB ProtoFiles ${CMAKE_CURRENT_LIST_DIR}/memory_manager/core/proto/*.proto)
protobuf_generate_cpp(ProtoSources ProtoHeaders ${ProtoFiles})


set(SRC_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/memory_manager)
set(SRC_DIR
    ${SRC_ROOT_PATH}/core
    ${SRC_ROOT_PATH}/core/proto # protos
    ${SRC_ROOT_PATH}/utils
    ${SRC_ROOT_PATH}/platform
    ${SRC_ROOT_PATH}/test
    )

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})


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


add_executable(memory ${SRC_FILES} ${ProtoSources})
target_link_libraries(memory -pthread ${Protobuf_LIBRARIES})
