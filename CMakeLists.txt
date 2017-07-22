cmake_minimum_required(VERSION 3.5)
project(mvshape)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_COMPILER ${CMAKE_SOURCE_DIR}/third_party/nix_env/bin/gcc)
set(CMAKE_CXX_COMPILER ${CMAKE_SOURCE_DIR}/third_party/nix_env/bin/g++)

add_compile_options(-nostdinc++ -nostdinc)
include_directories(BEFORE SYSTEM
        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include/c++/6.4.0
        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include/c++/6.4.0/x86_64-pc-linux-gnu
        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/gcc/x86_64-pc-linux-gnu/6.4.0/include
        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/gcc/x86_64-pc-linux-gnu/6.4.0/include-fixed
        )

# ---

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/nix_env)


include(third_party/CMakeLists.txt)


add_subdirectory(proto)



add_subdirectory(cpp)
