package(default_visibility = ["//visibility:public"])

eigen_header_files = glob([
        "Eigen/**",
        "unsupported/Eigen/**",
        ],
    exclude = [
        "Eigen/**/CMakeLists.txt",
        "unsupported/Eigen/**/CMakeLists.txt",
        ],
)

cc_library(
    name = "eigen3",
    hdrs = eigen_header_files,
    includes = ["."],
)
