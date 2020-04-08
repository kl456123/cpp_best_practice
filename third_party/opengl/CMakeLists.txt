cmake_minimum_required(VERSION 3.2.0)

project(opengl)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/examples/kernels
    ${CMAKE_SOURCE_DIR}/examples)

foreach(file_path ${SRC_DIR})
    file(GLOB file_source_c ${file_path}/*.cc)
    set(SRC_FILES ${SRC_FILES} ${file_source_c})
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/examples)
# message(${SRC_FILES})

add_executable(main ${SRC_FILES})

target_link_libraries(main -lglut -lGL -lGLEW)
