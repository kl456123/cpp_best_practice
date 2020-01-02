file(GLOB_RECURSE cpp_source_file "*.cpp")
file(GLOB_RECURSE cc_source_file "*.cc")
file(GLOB_RECURSE c_source_file "*.c")

set(Files ${cpp_source_file} ${cc_source_file} ${c_source_file})

include_directories(.)

add_executable(testsuite ${Files})


target_link_libraries(testsuite DLCL)
