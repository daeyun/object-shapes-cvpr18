package(default_visibility = ["//visibility:public"])

# Run the build scripts first.

cc_library(
    name = "opencv",
    srcs = glob([
        "install/opencv/lib/*.a",
        "install/opencv/share/OpenCV/**/*.a",
    ]),
    hdrs = glob(["install/opencv/include/**"]),
    includes = ["install/opencv/include"],
    linkopts = [
        "-lstdc++",
        "-ldl",
        "-lm",
        "-pthread",
        "-lz",
    ],
    linkstatic = 1,
)

# Uses namespace "google" instead of "gflags".
cc_library(
    name = "gflags",
    srcs = glob([
        "install/gflags/lib/libgflags.a",
    ]),
    hdrs = glob(["install/gflags/include/**"]),
    includes = ["install/gflags/include"],
    linkopts = [
        "-pthread",
    ],
    linkstatic = 1,
    visibility = ["//visibility:private"],
)

cc_library(
    name = "glog",
    srcs = glob([
        "install/glog/lib/libglog.a",
    ]),
    hdrs = glob(["install/glog/include/**"]),
    includes = ["install/glog/include"],
    linkopts = [
        "-pthread",
    ],
    linkstatic = 1,
    deps = [":gflags"],
)

cc_library(
    name = "tensorflow",
    srcs = glob([
        "install/tensorflow/lib/libtensorflow_cc.so",
    ]),
    hdrs = glob(
        [
            "install/tensorflow/include/tf/**/*.h",
            "install/tensorflow/include/tf/**/*.hpp",
            "install/tensorflow/include/tf/third_party/**",
        ],
        exclude = [
            "install/tensorflow/include/tf/third_party/**/*.so",
            "install/tensorflow/include/tf/third_party/**/*.so.*",
        ],
    ),
    copts = [
        "-Wno-sign-compare",
        "-Iinstall/tensorflow/include/tf",
    ],
    includes = [
        "install/tensorflow/include/tf",
        "install/tensorflow/include/tf/bazel-genfiles",
    ],
    deps = [
        "//external:eigen3",
    ],
)

cc_library(
    name = "assimp",
    srcs = glob([
        "install/assimp/lib/libassimp.a",
    ]),
    hdrs = glob(["install/assimp/include/**"]),
    includes = ["install/assimp/include"],
    linkstatic = 1,
)
