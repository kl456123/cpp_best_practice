

set(NN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

set(NN_SRC_FNAMES conv2d.cc const.cc
    binary.cc batchnorm.cc relu.cc
    pool.cc flatten.cc gemm.cc clip.cc
    transpose.cc shape.cc gather.cc 
    unsqueeze.cc concat.cc reshape.cc)

foreach(file_path ${NN_SRC_FNAMES})
    set(NN_SRC_FILES ${NN_SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${file_path})
endforeach()

# build nn shared library
add_library(gl_nn SHARED ${NN_SRC_FILES})
target_link_libraries(gl_nn gl_core)

add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
set_target_properties(main PROPERTIES LINK_FLAGS -Wl,--no-as-needed)
target_link_libraries(main gl_nn)
