

find_package(GTest REQUIRED)

add_executable(test_main ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cc)
target_link_libraries(test_main gl_core
    ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
