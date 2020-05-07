cmake_minimum_required(VERSION 3.2.0)

project(opengl VERSION 1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++14")

option(BUILD_EXAMPLES "BUILD OPENGL EXAMPLES" ON)
option(BUILD_NN "BUILD NERTUAL NETWORK" ON)
option(BUILD_TEST "BUILD OPENGL TEST" ON)
option(BUILD_ARM "USE GLES" OFF)
option(BUILD_DEBUG "ENABLE DEBUG MODE" ON)

if(BUILD_ARM)
    add_definitions(-DARM_PLATFORM)
endif()

if(BUILD_DEBUG)
    add_definitions(-DENABLE_OPENGL_CHECK_ERROR)
endif()

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(Protobuf REQUIRED)

set(CORE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/core)
set(UTIL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/utils)

set(CORE_SRC_NAMES
    program.cc
    context.cc
    buffer.cc
    texture.cc
    session.cc
    fbo_session.cc
    kernel.cc
    kernel_registry.cc
    init.cc
    tensor.cc
    allocator.cc
    buffer_base.cc

    # some protos
    dlxnet.pb.cc
    )

set(UTIL_SRC_NAMES macros.cc util.cc protobuf.cc)

foreach(fname ${CORE_SRC_NAMES})
    set(CORE_SRC_FILES ${CORE_SRC_FILES} ${CORE_SRC_DIR}/${fname})
endforeach()

foreach(fname ${UTIL_SRC_NAMES})
    set(UTIL_SRC_FILES ${UTIL_SRC_FILES} ${UTIL_SRC_DIR}/${fname})
endforeach()

# root directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(OpenGL_LIB -lglfw)
if(BUILD_ARM)
    set(OpenGL_LIB ${OpenGL_LIB} -lGLESv2)
else()
    set(OpenGL_LIB ${OpenGL_LIB} -lGL)
endif()

set(GL_CORE gl_core)

# build core library
add_library(${GL_CORE} ${CORE_SRC_FILES} ${UTIL_SRC_FILES})
target_link_libraries(${GL_CORE} ${OpenGL_LIB} glog::glog ${Protobuf_LIBRARIES})


# examples demo
if (BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/examples)
endif()


# deeplearning framework using opengl backend
if (BUILD_NN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/nn)
endif()

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/test)
endif()




