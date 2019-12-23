
aux_source_directory(. MODULE_SRCS)


add_executable(test_pool ${MODULE_SRCS})


target_link_libraries(test_pool  pool)
