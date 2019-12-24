

file(GLOB_RECURSE Files "*.cpp")
include_directories(.)

add_executable(testsuite ${Files})


target_link_libraries(testsuite pool)
