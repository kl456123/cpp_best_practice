

# build eigen
find_package(Eigen3 REQUIRED NO_MODULE)


aux_source_directory(Eigen test_eigen_src)

add_executable(test_eigen ${test_eigen_src})

target_link_libraries(test_eigen Eigen3::Eigen)
