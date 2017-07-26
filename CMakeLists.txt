#set(CMAKE_C_COMPILER ${CMAKE_SOURCE_DIR}/third_party/nix_env/bin/gcc)
#set(CMAKE_CXX_COMPILER ${CMAKE_SOURCE_DIR}/third_party/nix_env/bin/g++)

# ---

cmake_minimum_required(VERSION 3.5)
project(mvshape CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Threads REQUIRED)

#set(CMAKE_C_FLAGS "-nostdinc++ -nostdinc")
#set(CMAKE_CXX_FLAGS "-nostdinc++ -nostdinc")

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed,--no-undefined,--error-unresolved-symbols")
set(CMAKE_EXEC_LINKER_FLAGS "-Wl,--as-needed,--no-undefined,--error-unresolved-symbols")

#SET(CMAKE_SKIP_BUILD_RPATH FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#SET(CMAKE_INSTALL_RPATH
#        $ORIGIN
#        /usr/lib/nvidia-381
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib64/
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/gcc/x86_64-pc-linux-gnu/6.4.0/
#        ${CMAKE_BINARY_DIR}/proto/
#        $ORIGIN/../egl/
#        /usr/lib/x86_64-linux-gnu/
#        /lib/x86_64-linux-gnu/
#        )
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

#include_directories(BEFORE
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include/c++/6.4.0
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include/c++/6.4.0/x86_64-pc-linux-gnu
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/gcc/x86_64-pc-linux-gnu/6.4.0/include
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/lib/gcc/x86_64-pc-linux-gnu/6.4.0/include-fixed
#        )


# ---

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/nix_env)


include(third_party/CMakeLists.txt)

add_subdirectory(proto)

add_subdirectory(cpp)
