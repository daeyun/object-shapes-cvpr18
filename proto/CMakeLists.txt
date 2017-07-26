find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_SOURCE_DIR}/proto/dataset.proto)
PROTOBUF_GENERATE_PYTHON(PROTO_PY ${CMAKE_SOURCE_DIR}/proto/dataset.proto)

add_library(mvshape_proto SHARED ${PROTO_SRC} ${PROTO_HEADER})
target_link_libraries(mvshape_proto PUBLIC ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/libprotobuf.so.12)
target_include_directories(mvshape_proto PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(mvshape_proto PUBLIC ${CMAKE_BINARY_DIR})
