

find_package(GTest REQUIRED)

# opengl core test
file(GLOB_RECURSE TEST_CORE_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/core/*_test.cc)

# nn ops test
file(GLOB_RECURSE TEST_NN_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/nn/kernels/*.cc)

add_executable(test_main ${TEST_CORE_SRC_FILES} ${TEST_NN_SRC_FILES})
target_link_libraries(test_main ${GL_CORE} ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES})
