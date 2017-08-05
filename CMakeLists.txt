cmake_minimum_required(VERSION 3.5)
project(mvshape CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Threads REQUIRED)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed,--no-undefined,--error-unresolved-symbols")
set(CMAKE_EXEC_LINKER_FLAGS "-Wl,--as-needed,--no-undefined,--error-unresolved-symbols")

set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=/usr/local/cuda-8.0/lib64")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -mfpmath=both")


find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# ---

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")

add_subdirectory(third_party)

add_subdirectory(proto)
add_subdirectory(cpp)
