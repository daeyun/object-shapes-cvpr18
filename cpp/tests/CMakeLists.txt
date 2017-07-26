find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_library(test_main test_main.cc)
target_include_directories(test_main SYSTEM PUBLIC
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp/lib
        )
target_link_libraries(test_main PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GTEST_LIBRARIES}
        mvshape_cpp_core
        )

# ---

add_executable(runAllTests
        #        file_io_test.cc
        #        resources_test.cc
                egl_rendering_test.cc
        #        string_utils_test.cc
        #        database_test.cc
        #        dataset_io_test.cc
        #        camera_test.cc
        #        multiprocessing_test.cc
        #        random_utils_test.cc
#        tf_utils_test.cc
#        tensorflow_test.cc
        )
target_link_libraries(runAllTests
        pthread
        m
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GTEST_LIBRARIES}
        test_main
        mvshape_cpp_core
#        tensorflow
#        tf_utils
        mvshape_proto
        #        data_io
        #        tf_utils
        #        database
        egl_rendering
        )

#add_executable(eglTest
#        egl_test.cc
#        )
#target_link_libraries(eglTest
#        ${CMAKE_THREAD_LIBS_INIT}
#        ${GTEST_BOTH_LIBRARIES}
#        hello
#        )
#target_include_directories(eglTest PUBLIC
#        ${CMAKE_SOURCE_DIR}
#        )
#add_executable(eglTest
#        egl_test_app.cc
#        )
#target_link_libraries(eglTest
##        ${CMAKE_THREAD_LIBS_INIT}
##        ${GTEST_BOTH_LIBRARIES}
#        hello
#        )
#target_include_directories(eglTest PUBLIC
#        ${CMAKE_SOURCE_DIR}
#        )
