add_library(mvshape_cpp_core STATIC
        common.cc
        file_io.cc
        benchmark.cc
        flags.cc
        string_utils.cc
        resources.cc
        multiprocessing.cc
        transforms.cc
        camera.cc
        random_utils.cc
        batch_queue.cc
        common.h
        file_io.h
        benchmark.h
        flags.h
        string_utils.h
        resources.h
        multiprocessing.h
        transforms.h
        camera.h
        random_utils.h
        batch_queue.h
        )
target_include_directories(mvshape_cpp_core PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib)
target_link_libraries(mvshape_cpp_core PUBLIC
        third_party_core
        )

#add_library(egl_rendering STATIC egl_rendering.cc)
#target_link_libraries(egl_rendering PUBLIC mvshape_cpp_core /home/daeyun/tmp/libegl_renderer.so)

#add_library(data_io STATIC data_io.cc)
#target_link_libraries(data_io mvshape_cpp_core mvshape_proto egl_rendering)

#add_library(tf_utils STATIC tf_utils.cc)
#target_link_libraries(tf_utils mvshape_cpp_core tensorflow)

#add_library(database STATIC database.cc)
#target_link_libraries(database mvshape_cpp_core)

#add_library(mvshape_helpers STATIC mvshape_helpers.cc)
#target_link_libraries(mvshape_helpers mvshape_cpp_core database tf_utils data_io)
