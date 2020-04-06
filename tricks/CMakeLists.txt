cmake_minimum_required(VERSION 3.2)

project(absl_demo)



set(LIB_PATH /usr/local/lib)
file(GLOB ABSL_LIBS ${LIB_PATH}/libabsl_*.a)

# foreach(file_path ${ABSL_LIBS_FN})
    # set(ABSL_LIBS ${ABSL_LIBS} -l${file_path})
# endforeach()

add_executable(absl_demo absl.cc)

target_link_libraries(absl_demo ${ABSL_LIBS})

add_executable(scanner_demo scanner.cc)

target_link_libraries(scanner_demo ${ABSL_LIBS})
