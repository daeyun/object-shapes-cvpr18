package(default_visibility = ["//visibility:public"])

cc_library(
    name = "blosc",
    srcs = glob(
        ["blosc/*.c"],
        exclude = ["blosc/*-avx2.c"],
    ),
    hdrs = glob(
        ["blosc/*.h"],
        exclude = ["blosc/*-avx2.h"],
    ),
    copts = [
        "-march=native",
        "-O3",
        "-g",
    ],
    includes = ["blosc/"],
)
