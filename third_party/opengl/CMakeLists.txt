cmake_minimum_required(VERSION 3.2.0)

project(opengl)

file(GLOB SRC ${CMAKE_SOURCE_DIR}/examples/*.cc)

add_executable(main ${SRC})


target_link_libraries(main -lglut -lGL -lGLEW)
