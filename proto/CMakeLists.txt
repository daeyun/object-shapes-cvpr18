# Needs to make sure the Nix symlinks are accessible.
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/install/protobuf)
find_package(Protobuf 3.2.0 EXACT REQUIRED)

message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}")

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_SOURCE_DIR}/proto/dataset.proto)
PROTOBUF_GENERATE_PYTHON(PROTO_PY ${CMAKE_SOURCE_DIR}/proto/dataset.proto)

add_library(mvshape_proto SHARED ${PROTO_SRC} ${PROTO_HEADER})
target_link_libraries(mvshape_proto PUBLIC ${Protobuf_LIBRARIES})
target_include_directories(mvshape_proto PUBLIC ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(mvshape_proto PUBLIC ${CMAKE_BINARY_DIR})

