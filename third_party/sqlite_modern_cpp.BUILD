package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sqlite_modern_cpp",
    hdrs = glob(
        ["hdr/**"],
    ),
    include_prefix = "sqlite_modern_cpp",
    includes = ["hdr"],
    strip_include_prefix = "hdr",
    deps = ["@//third_party:sqlite3"]
)
