#set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed to build shared objects for python ctypes.

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
target_include_directories(mvshape_cpp_core BEFORE PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib
        ${CMAKE_BINARY_DIR}  # for proto/*
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        )
target_link_libraries(mvshape_cpp_core PUBLIC
        third_party_core
        )

add_library(egl_rendering STATIC egl_rendering.cc)
target_link_libraries(egl_rendering
        PUBLIC
        mvshape_cpp_core
        PRIVATE
        egl)



add_library(voxel STATIC
        common.cc common.h
        voxel.cc voxel.h
        )
#set_property(TARGET voxel PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(voxel BEFORE PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib
        ${CMAKE_SOURCE_DIR}/third_party/src/polygon2voxel
        ${CMAKE_SOURCE_DIR}/third_party/repos/cpp_gsl/include
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        )
target_link_libraries(voxel PUBLIC
        polygon2voxel
        )



add_library(data_io STATIC data_io.cc)
target_link_libraries(data_io mvshape_cpp_core mvshape_proto egl_rendering database voxel)

#add_library(tf_utils STATIC tf_utils.cc)
#target_link_libraries(tf_utils PUBLIC mvshape_cpp_core tensorflow)
#target_include_directories(tf_utils
#        PUBLIC
#        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
#        ${CMAKE_SOURCE_DIR}/third_party/install/protobuf/include
#        )

add_library(database STATIC database.cc)
target_link_libraries(database mvshape_cpp_core)

#add_library(mvshape_helpers STATIC mvshape_helpers.cc)
#target_link_libraries(mvshape_helpers mvshape_cpp_core database
#        tf_utils
#        data_io)

add_library(fssr_utils STATIC fssr_utils.cc)
target_link_libraries(fssr_utils mvshape_cpp_core mve data_io)

add_library(meshdist STATIC meshdist.cc)
target_link_libraries(meshdist mvshape_cpp_core mve data_io)



find_package(PNG REQUIRED)
find_package(BLOSC REQUIRED)
add_library(single_batch_loader STATIC
        common.cc
        single_batch_loader.cc single_batch_loader.h
        )
set_property(TARGET single_batch_loader PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(single_batch_loader BEFORE PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib
        ${CMAKE_BINARY_DIR}  # for proto/*
        ${CMAKE_SOURCE_DIR}/third_party/repos/eigen
        ${CMAKE_SOURCE_DIR}/third_party/repos/cpp_gsl/include
        ${PNG_INCLUDE_DIR}
        ${BLOSC_INCLUDE_DIR}
        )
target_link_libraries(single_batch_loader PUBLIC
        ${PNG_LIBRARY}
        ${BLOSC_LIBRARIES}
        )
