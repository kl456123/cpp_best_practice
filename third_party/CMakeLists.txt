

# build eigen
find_package(Eigen3 REQUIRED NO_MODULE)


aux_source_directory(Eigen test_eigen_src)

add_executable(test_eigen ${test_eigen_src})

target_link_libraries(test_eigen Eigen3::Eigen)


# build test_protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB ProtoFiles ${CMAKE_CURRENT_LIST_DIR}/protos/*.proto)
protobuf_generate_cpp(ProtoSources ProtoHeaders ${ProtoFiles})

add_executable(test_proto ${ProtoSources} ${ProtoHeaders} ./protos/test_proto.cpp)
target_link_libraries(test_proto ${Protobuf_LIBRARIES})
