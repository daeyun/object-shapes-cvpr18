cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//external:cpp_gsl",
        "//external:eigen3",
    ],
)

cc_library(
    name = "camera",
    srcs = ["camera.cc"],
    hdrs = ["camera.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    hdrs = ["benchmark.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "data_io",
    srcs = ["data_io.cc"],
    hdrs = ["data_io.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch_queue",
        ":benchmark",
        ":camera",
        ":common",
        ":database",
        ":egl_rendering",
        ":file_io",
        ":multiprocessing",
        ":random_utils",
        ":transforms",
        "//external:cpp_lru_cache",
        "//proto:dataset_cc_proto",
    ],
)

cc_library(
    name = "database",
    srcs = ["database.cc"],
    hdrs = ["database.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":file_io",
        ":flags",
        "//external:sqlite_modern_cpp",
        "//proto:dataset_cc_proto",
    ],
)

cc_library(
    name = "egl_rendering",
    srcs = [
        "egl_rendering.cc",
    ],
    hdrs = ["egl_rendering.h"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":camera",
        ":common",
        ":file_io",
        ":resources",
        "//third_party:assimp",
        "//third_party:gflags",
        "//third_party:glog",
        "//third_party:opencv",
    ],
)

# This should be small.
cc_library(
    name = "file_io",
    srcs = [
        "file_io.cc",
    ],
    hdrs = ["file_io.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark",
        ":common",
        ":flags",
        "//third_party:assimp",
        "//third_party:blosc",
        "//third_party:gflags",
        "//third_party:glog",
        "//third_party:opencv",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:gflags",
    ],
)

cc_library(
    name = "resources",
    srcs = ["resources.cc"],
    hdrs = ["resources.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":file_io",
        ":flags",
        "//third_party:gflags",
        "//third_party:glog",
    ],
)

cc_library(
    name = "multiprocessing",
    srcs = ["multiprocessing.cc"],
    hdrs = ["multiprocessing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//external:cpp_gsl",
        "//third_party:gflags",
        "//third_party:glog",
    ],
)

cc_library(
    name = "random_utils",
    srcs = ["random_utils.cc"],
    hdrs = ["random_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "string_utils",
    srcs = ["string_utils.cc"],
    hdrs = ["string_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "@boost//:algorithm",
    ],
)

cc_library(
    name = "transforms",
    srcs = ["transforms.cc"],
    hdrs = ["transforms.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":file_io",
        ":string_utils",
        "//third_party:gflags",
        "//third_party:glog",
    ],
)

cc_library(
    name = "tf_utils",
    srcs = ["tf_utils.cc"],
    hdrs = ["tf_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":flags",
        ":resources",
        ":string_utils",
        "//proto:dataset_cc_proto",
        "//third_party:gflags",
        "//third_party:glog",
        "//third_party:tensorflow",
        "@boost//:algorithm",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "batch_queue",
    srcs = ["batch_queue.cc"],
    hdrs = ["batch_queue.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera",
        ":common",
        ":file_io",
        "//external:cqueue",
    ],
)

cc_library(
    name = "mvshape_helpers",
    srcs = ["mvshape_helpers.cc"],
    hdrs = ["mvshape_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark",
        ":common",
        ":data_io",
        ":database",
        ":file_io",
        ":random_utils",
        ":string_utils",
        ":tf_utils",
        "//external:gflags",
        "//third_party:blosc",
        "//third_party:glog",
        "//third_party:tensorflow",
    ],
)

cc_library(
    name = "voxel",
    srcs = ["voxel.cc"],
    hdrs = ["voxel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark",
        ":camera",
        ":common",
        ":data_io",
        ":file_io",
        ":random_utils",
        ":string_utils",
        "//external:gflags",
        "//third_party:glog",
        "//third_party/src/polygon2voxel",
    ],
)
