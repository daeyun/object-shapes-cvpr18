find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(GTest REQUIRED)

add_library(test_main test_main.cc)
target_include_directories(test_main SYSTEM PUBLIC
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib
        )
target_link_libraries(test_main PUBLIC
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GTEST_LIBRARIES}
        mvshape_cpp_core
        )

# ---

add_executable(runAllTests
        file_io_test.cc
        resources_test.cc
        egl_rendering_test.cc
        string_utils_test.cc
        database_test.cc
        dataset_io_test.cc
        camera_test.cc
        multiprocessing_test.cc
        random_utils_test.cc
        tf_utils_test.cc
        benchmark_test.cc
        fssr_utils_test.cc
        single_batch_loader_test.cc
        test_utils.h
        test_utils.cc
        )

target_link_libraries(runAllTests
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GTEST_LIBRARIES}
        test_main
        mvshape_cpp_core
        tensorflow
        tf_utils
        mvshape_proto
        data_io
        database
        egl_rendering
        single_batch_loader
        fssr_utils
        )

