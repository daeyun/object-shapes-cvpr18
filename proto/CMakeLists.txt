find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_SOURCE_DIR}/proto/dataset.proto)
PROTOBUF_GENERATE_PYTHON(PROTO_PY ${CMAKE_SOURCE_DIR}/proto/dataset.proto)

file(GLOB PROTO_FILES ${CMAKE_BINARY_DIR}/proto/*.pb.*)
add_library(mvshape_proto SHARED ${PROTO_FILES})
target_link_libraries(mvshape_proto ${PROTOBUF_LIBRARIES})
target_include_directories(mvshape_proto PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(mvshape_proto PUBLIC ${CMAKE_BINARY_DIR})
