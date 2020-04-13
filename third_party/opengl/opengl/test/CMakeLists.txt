

find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/*_test.cc)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)


add_executable(test_main ${SRC_FILES} ${TEST_SRC_FILES})
target_link_libraries(test_main ${GL_CORE} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
