set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake/Modules/")
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/nix_env)

#set(BUILD_SHARED_LIBS OFF)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ASSIMP REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLOSC REQUIRED)

add_library(third_party_core INTERFACE)
target_include_directories(third_party_core INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/repos/cpp_gsl/include
        ${CMAKE_CURRENT_LIST_DIR}/repos/eigen
        ${CMAKE_CURRENT_LIST_DIR}/nix_env/include
        ${CMAKE_CURRENT_LIST_DIR}/repos/sqlite_modern_cpp/hdr
        ${CMAKE_CURRENT_LIST_DIR}/repos/concurrentqueue
        ${CMAKE_CURRENT_LIST_DIR}/repos/cpp-lru-cache/include
        ${GLESv2_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIR}
        ${ASSIMP_INCLUDE_DIR}
        ${BLOSC_INCLUDE_DIR}
        ${SQLITE3_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(third_party_core INTERFACE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_LOG_LIBRARY}

        # ${GFLAGS_LIBRARIES}
         ${SQLITE3_LIBRARY_RELEASE}
         ${ASSIMP_LIBRARIES}
         ${GLOG_LIBRARIES}

        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libgflags.a
        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libblosc.a

        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libz.so
        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libprotobuf.so
        )


add_library(tensorflow INTERFACE)
target_include_directories(tensorflow SYSTEM INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf/bazel-genfiles
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf/third_party/eigen3
        )
target_link_libraries(tensorflow INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/lib/libtensorflow_cc.so
        )

#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/links/usr/include/)
#ADD_CUSTOM_TARGET(egl_symlink
#        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/EGL ${CMAKE_BINARY_DIR}/third_party/links/usr/include/EGL
#        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/GLES3 ${CMAKE_BINARY_DIR}/third_party/links/usr/include/GLES3
#        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/KHR ${CMAKE_BINARY_DIR}/third_party/links/usr/include/KHR
#        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/X11 ${CMAKE_BINARY_DIR}/third_party/links/usr/include/X11
#        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/GL ${CMAKE_BINARY_DIR}/third_party/links/usr/include/GL
#        )

#add_library(egl INTERFACE)
#target_link_libraries(egl INTERFACE
#        ${OPENGL_LIBRARIES}
#        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libz.so
##        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libEGL.so
##        ${CMAKE_CURRENT_LIST_DIR}/nix_env/lib/libGLESv2.so
#        /usr/lib/nvidia-381/libEGL.so
#        /usr/lib/nvidia-381/libGLESv2.so
#        /usr/lib/nvidia-381/libGLdispatch.so.0
#        )
#target_include_directories(egl SYSTEM INTERFACE
#        ${EGL_INCLUDE_DIR}
#        ${CMAKE_SOURCE_DIR}/third_party/nix_env/include
#        ${CMAKE_BINARY_DIR}/third_party/links/usr/include
#        )
##add_dependencies(egl egl_symlink)
