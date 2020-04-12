cmake_minimum_required(VERSION 3.2.0)

option(BUILD_DEMO "BUILD OPENGL DEMO" OFF)
option(BUILD_GPGPU "BUILD OPENGL GPGPU" ON)

project(opengl)

find_package(OpenCV REQUIRED)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/examples/kernels
    ${CMAKE_SOURCE_DIR}/examples)

foreach(file_path ${SRC_DIR})
    file(GLOB file_source_c ${file_path}/*.cc)
    set(SRC_FILES ${SRC_FILES} ${file_source_c})
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/examples)
# message(${SRC_FILES})

set(OpenGL_LIB -lglut -lGL -lGLEW -lglfw)
add_executable(main ${SRC_FILES})

target_link_libraries(main ${OpenGL_LIB} ${OpenCV_LIBS})


if(BUILD_DEMO)
# build demo
    add_executable(demo ${CMAKE_SOURCE_DIR}/demo/demo.cc)
    target_link_libraries(demo ${OpenGL_LIB} ${OpenCV_LIBS})
endif()

if(BUILD_GPGPU)
    add_executable(axpy_demo ${CMAKE_SOURCE_DIR}/gpgpu/axpy_demo.cc)
    target_link_libraries(axpy_demo ${OpenGL_LIB} ${OpenCV_LIBS})

    add_executable(compute_shader ${CMAKE_SOURCE_DIR}/gpgpu/compute_shader.cc)
    target_link_libraries(compute_shader ${OpenGL_LIB} ${OpenCV_LIBS})
endif()
