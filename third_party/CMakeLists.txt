add_library(cpp_gsl INTERFACE)
target_include_directories(cpp_gsl SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/repos/cpp_gsl/include)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include/eigen3)

add_library(opencv3 INTERFACE)
target_include_directories(opencv3 SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(glog INTERFACE)
target_include_directories(glog SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(cqueue INTERFACE)
target_include_directories(cqueue SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/repos/concurrentqueue)

add_library(gflags INTERFACE)
target_include_directories(gflags SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(boost INTERFACE)
target_include_directories(boost SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(assimp INTERFACE)
target_include_directories(assimp SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(blosc INTERFACE)
target_include_directories(blosc SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(tensorflow INTERFACE)
target_include_directories(tensorflow SYSTEM INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf/bazel-genfiles
        ${CMAKE_CURRENT_LIST_DIR}/install/tensorflow/include/tf/third_party/eigen3
        )

add_library(sqlite3 INTERFACE)
target_include_directories(sqlite3 SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nix_env/include)

add_library(sqlite_modern_cpp INTERFACE)
target_include_directories(sqlite_modern_cpp SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/repos/sqlite_modern_cpp/hdr)
target_link_libraries(sqlite_modern_cpp INTERFACE sqlite3)

add_library(lrucache INTERFACE)
target_include_directories(lrucache SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/repos/cpp-lru-cache/include)

find_package(EGL)
find_package(GLESv2)
#add_library(egl INTERFACE)
#target_include_directories(egl SYSTEM INTERFACE ${EGL_INCLUDE_DIR} /usr/include)
#target_link_libraries(egl INTERFACE ${EGL_LIBRARY})
#set_target_properties(egl PROPERTIES COMPILE_FLAGS "-isystem /usr/include")

ADD_CUSTOM_TARGET(egl_symlink
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/EGL ${CMAKE_BINARY_DIR}/third_party/EGL
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/GLES3 ${CMAKE_BINARY_DIR}/third_party/GLES3
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/KHR ${CMAKE_BINARY_DIR}/third_party/KHR
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/X11 ${CMAKE_BINARY_DIR}/third_party/X11
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${EGL_INCLUDE_DIR}/GL ${CMAKE_BINARY_DIR}/third_party/GL
        )

add_library(egl INTERFACE)
target_link_libraries(egl INTERFACE ${EGL_LIBRARY})
target_include_directories(egl SYSTEM INTERFACE ${CMAKE_BINARY_DIR}/third_party)
add_dependencies(egl egl_symlink)
