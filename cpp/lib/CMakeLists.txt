add_library(common STATIC common.cc)
target_link_libraries(common cpp_gsl eigen3 glog gflags)

add_library(random_utils STATIC random_utils.cc)
target_link_libraries(random_utils common)

add_library(camera STATIC camera.cc)
target_link_libraries(camera common)

add_library(transforms STATIC transforms.cc)
target_link_libraries(transforms common opencv3)

add_library(benchmark STATIC benchmark.cc)
target_link_libraries(benchmark common)

add_library(batch_queue STATIC batch_queue.cc)
target_link_libraries(batch_queue common cqueue)

add_library(flags STATIC flags.cc)
target_link_libraries(flags common)

add_library(multiprocessing STATIC multiprocessing.cc)
target_link_libraries(multiprocessing common glog)

add_library(resources STATIC resources.cc)
target_link_libraries(resources common boost)

add_library(file_io STATIC file_io.cc)
target_link_libraries(file_io common boost assimp blosc opencv3)

add_library(database STATIC database.cc)
target_link_libraries(database common boost mvshape_proto sqlite_modern_cpp)

add_library(string_utils STATIC string_utils.cc)
target_link_libraries(string_utils common boost)

add_library(egl_rendering STATIC egl_rendering.cc)
target_link_libraries(egl_rendering common boost opencv3 egl)

add_library(data_io STATIC data_io.cc)
target_link_libraries(data_io common batch_queue boost assimp blosc opencv3 lrucache mvshape_proto egl_rendering)

add_library(tf_utils STATIC tf_utils.cc)
target_link_libraries(tf_utils common file_io string_utils boost tensorflow mvshape_proto)

