

set(NN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

foreach(file_path ${NN_SRC_DIR})
    file(GLOB file_source_c ${file_path}/*.cc)
    set(NN_SRC_FILES ${NN_SRC_FILES} ${file_source_c})
endforeach()

add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cc ${NN_SRC_FILES})
target_link_libraries(main ${GL_CORE})
