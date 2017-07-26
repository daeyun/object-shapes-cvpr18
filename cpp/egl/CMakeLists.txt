# reset prefix and flags
#unset(CMAKE_PREFIX_PATH)
#unset(CMAKE_C_FLAGS)
#unset(CMAKE_CXX_FLAGS)
#unset(CMAKE_SHARED_LINKER_FLAGS)
#unset(CMAKE_EXEC_LINKER_FLAGS)


# reset rpath, enable building with rpath
#SET(CMAKE_SKIP_BUILD_RPATH FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#SET(CMAKE_INSTALL_RPATH
#        $ORIGIN
#        /usr/lib/nvidia-381
#        /usr/lib/i386-linux-gnu/
#        )
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#SET(CMAKE_PREFIX_PATH ${CMAKE_SYSTEM_PREFIX_PATH})
#
## reset compiler
#unset(CMAKE_CXX_COMPILER)
#unset(CMAKE_C_COMPILER)
#find_program(CMAKE_C_COMPILER gcc)
#find_program(CMAKE_CXX_COMPILER g++)
#
## Reset include_directories
#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")




###########################################
#project(mvshape CXX)
#
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_VERBOSE_MAKEFILE ON)


# find packages

find_path(EGL_SYSTEM_INCLUDE_DIR EGL/egl.h PATHS
        /usr/local/include
        /usr/include
        )
find_library(EGL_SYSTEM_LIBRARY EGL PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        )
if (NOT EGL_SYSTEM_LIBRARY)
    message(FATAL_ERROR "Error: library EGL not found.")
endif ()

find_path(GLESv2_SYSTEM_INCLUDE_DIR GLES2/gl2.h PATHS
        /usr/local/include
        /usr/include
        )
find_library(GLESv2_SYSTEM_LIBRARY GLESv2 PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        )
if (NOT GLESv2_SYSTEM_LIBRARY)
    message(FATAL_ERROR "Error: library GLESv2 not found.")
endif ()

message(STATUS "EGL_SYSTEM_INCLUDE_DIR: ${EGL_SYSTEM_INCLUDE_DIR}")
message(STATUS "GLESv2_SYSTEM_INCLUDE_DIR: ${GLESv2_SYSTEM_INCLUDE_DIR}")
message(STATUS "EGL_SYSTEM_LIBRARY: ${EGL_SYSTEM_LIBRARY}")

########################


add_library(egl_renderer SHARED egl_renderer.cc egl_renderer.h)
target_include_directories(egl_renderer
        PRIVATE
        ${EGL_SYSTEM_INCLUDE_DIR}
        ${GLESv2_SYSTEM_INCLUDE_DIR}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        )
target_link_libraries(egl_renderer
        PRIVATE
        ${EGL_SYSTEM_LIBRARY}
        ${GLESv2_SYSTEM_LIBRARY}
        )

#find_package(GFlags REQUIRED)
#find_package(GTest REQUIRED)
#find_package(Threads REQUIRED)
#
add_library(hello SHARED hello.cc)
target_include_directories(hello PUBLIC
        ${EGL_SYSTEM_INCLUDE_DIR}
        ${GLESv2_SYSTEM_INCLUDE_DIR}
        )
target_link_libraries(hello
        PRIVATE
        ${EGL_SYSTEM_LIBRARY}
        ${GLESv2_SYSTEM_LIBRARY}
        )

add_executable(hello_main hello_test.cc)
target_include_directories(hello_main PUBLIC
        ${CMAKE_SOURCE_DIR}
        )
target_link_libraries(hello_main
#        ${GFLAGS_LIBRARIES}
#        ${GTEST_BOTH_LIBRARIES}
        hello
        )

