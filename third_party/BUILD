package(default_visibility = ["//visibility:public"])

# Run the build scripts first.

cc_library(
    name = "opencv",
    srcs = glob([
        "install/opencv/lib/*.so",
        "install/opencv/lib/*.so.*",
    ]),
    hdrs = glob(["install/opencv/include/**"]),
    includes = ["install/opencv/include"],
)

cc_library(
    name = "opencv_static",

    # Order matters. See pkgconfig/opencv.pc
    srcs = [
        "install/opencv/lib/libopencv_highgui.a",
        "install/opencv/lib/libopencv_imgcodecs.a",
        "install/opencv/lib/libopencv_imgproc.a",
        "install/opencv/lib/libopencv_core.a",
    ] + glob([
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
        "-ljpeg",
        "-lpng",
        "-lrt",
    ],
)

# Uses namespace "google" instead of "gflags".
cc_library(
    name = "gflags",
    srcs = glob(["install/gflags/lib/libgflags.so*"]),
    hdrs = glob(["install/gflags/include/**"]),
    includes = ["install/gflags/include"],
    linkopts = [
        "-pthread",
    ],
    #linkstatic = 1,
    #visibility = ["//visibility:private"],
)

cc_library(
    name = "glog",
    srcs = ["install/glog/lib/libglog.a"],
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
    srcs = ["install/tensorflow/lib/libtensorflow_cc.so"],
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
        #"install/assimp/lib/libassimp.a",
        "install/assimp/lib/libassimp.so*",
    ]),
    hdrs = glob(["install/assimp/include/**"]),
    includes = ["install/assimp/include"],
    linkopts = [
        "-lstdc++",
        "-lz",
    ],
    #linkstatic = 1,
)

cc_library(
    name = "sqlite3",
    srcs = [
        "install/sqlite3/lib/libsqlite3.a",
    ] + glob([
        "install/sqlite3/lib/libsqlite3.so*",
    ]),
    hdrs = glob(["install/sqlite3/include/**"]),
    includes = ["install/sqlite3/include"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    linkstatic = 1,
)

cc_library(
    name = "blosc",
    srcs = glob([
        #"install/c-blosc/lib/libblosc.a",
        "install/c-blosc/lib/libblosc.so*",
    ]),
    hdrs = glob(["install/c-blosc/include/**"]),
    includes = ["install/c-blosc/include"],
    #linkstatic = 1,
)
