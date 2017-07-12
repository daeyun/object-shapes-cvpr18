cc_library(
    name = "test_main",
    srcs = ["test_main.cc"],
    copts = ["-Icpp/lib"],
    data = ["//resources"],
    deps = [
        "//cpp/lib:common",
        "//cpp/lib:file_io",
        "//cpp/lib:flags",
        "//cpp/lib:resources",
        "//external:cpp_gsl",
        "//external:gtest",
        "//third_party:gflags",
        "//third_party:glog",
    ],
)

cc_test(
    name = "camera_test",
    size = "small",
    srcs = ["camera_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:camera",
    ],
)

cc_test(
    name = "database_test",
    size = "small",
    srcs = ["database_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:database",
    ],
)

cc_test(
    name = "dataset_io_test",
    size = "small",
    srcs = ["dataset_io_test.cc"],
    copts = ["-Icpp/lib"],
    # Tests should run one at a time.
    tags = ["exclusive"],
    deps = [
        ":test_main",
        "//cpp/lib:data_io",
        "//cpp/lib:database",
    ],
)

cc_test(
    name = "egl_rendering_test",
    size = "small",
    srcs = ["egl_rendering_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:egl_rendering",
        "//external:eigen3",
    ],
)

cc_test(
    name = "file_io_test",
    size = "small",
    srcs = ["file_io_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:file_io",
    ],
)

cc_test(
    name = "multiprocessing_test",
    size = "small",
    srcs = ["multiprocessing_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:multiprocessing",
    ],
)

cc_test(
    name = "random_utils_test",
    size = "small",
    srcs = ["random_utils_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:random_utils",
    ],
)

cc_test(
    name = "resources_test",
    size = "small",
    srcs = ["resources_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:resources",
    ],
)

cc_test(
    name = "string_utils_test",
    size = "small",
    srcs = ["string_utils_test.cc"],
    copts = ["-Icpp/lib"],
    deps = [
        ":test_main",
        "//cpp/lib:string_utils",
    ],
)
