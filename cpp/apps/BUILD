cc_binary(
    name = "generate_dataset",
    srcs = ["generate_dataset.cc"],
    includes = ["cpp/lib"],
    deps = [
        "//cpp/lib:data_io",
        "//cpp/lib:database",
        "//cpp/lib:egl_rendering",
        "//cpp/lib:flags",
        "//cpp/lib:resources",
        "//cpp/lib:string_utils",
        "//cpp/lib:transforms",
        "//external:gflags",
        "//third_party:opencv",
    ],
)

cc_binary(
    name = "train",
    srcs = [
        "train.cc",
    ],
    deps = [
        "//cpp/lib:benchmark",
        "//cpp/lib:data_io",
        "//cpp/lib:database",
        "//cpp/lib:file_io",
        "//cpp/lib:mvshape_helpers",
        "//cpp/lib:random_utils",
        "//cpp/lib:string_utils",
        "//cpp/lib:tf_utils",
        "//external:gflags",
        "//third_party:blosc",
        "//third_party:glog",
        "//third_party:tensorflow",
    ],
)
