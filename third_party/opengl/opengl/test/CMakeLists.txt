

find_package(GTest REQUIRED)

# opengl core test
file(GLOB_RECURSE TEST_CORE_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/core/*_test.cc)

# nn ops test
file(GLOB_RECURSE TEST_NN_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/nn/kernels/*_test.cc)

add_executable(test_main ${CMAKE_CURRENT_LIST_DIR}/test_main.cc
    ${TEST_CORE_SRC_FILES} ${TEST_NN_SRC_FILES})
set_target_properties(test_main PROPERTIES LINK_FLAGS -Wl,--no-as-needed)
target_link_libraries(test_main ${GL_NN} ${GTEST_LIBRARIES})
