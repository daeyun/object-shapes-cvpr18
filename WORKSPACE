http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-b4b0e304be5a68de3d0ee1af9b286f958750f5e4",
    urls = ["https://github.com/google/protobuf/archive/b4b0e30.zip"],
)

http_archive(
    name = "com_google_protobuf_cc",
    strip_prefix = "protobuf-b4b0e304be5a68de3d0ee1af9b286f958750f5e4",
    urls = ["https://github.com/google/protobuf/archive/b4b0e30.zip"],
)

new_http_archive(
    name = "eigen3_archive",
    build_file = "third_party/eigen3.BUILD",
    strip_prefix = "eigen-eigen-5a0156e40feb",
    url = "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz",
)

bind(
    name = "eigen3",
    actual = "@eigen3_archive//:eigen3",
)

new_http_archive(
    name = "cpp_gsl_archive",
    build_file = "third_party/cpp_gsl.BUILD",
    strip_prefix = "GSL-1f87ef73f1477e8adafa8b10ccee042897612a20",
    url = "https://github.com/Microsoft/GSL/archive/1f87ef7.tar.gz",
)

bind(
    name = "cpp_gsl",
    actual = "@cpp_gsl_archive//:cpp_gsl",
)

new_http_archive(
    name = "cqueue_archive",
    build_file = "third_party/cqueue.BUILD",
    strip_prefix = "concurrentqueue-5170da7fd04335da6d4e1d2ed7209a539176e589",
    url = "https://github.com/cameron314/concurrentqueue/archive/5170da7.tar.gz",
)

bind(
    name = "cqueue",
    actual = "@cqueue_archive//:cqueue",
)

new_http_archive(
    name = "cpp_lru_cache_archive",
    build_file = "third_party/cpp_lru_cache.BUILD",
    strip_prefix = "cpp-lru-cache-de1c4a03569bf3bd540e7f55ab5c2961411dbe22",
    urls = ["https://github.com/lamerman/cpp-lru-cache/archive/de1c4a03569bf3bd540e7f55ab5c2961411dbe22.zip"],
)

bind(
    name = "cpp_lru_cache",
    actual = "@cpp_lru_cache_archive//:cpp_lru_cache",
)

new_http_archive(
    name = "sqlite_modern_cpp_archive",
    build_file = "third_party/sqlite_modern_cpp.BUILD",
    strip_prefix = "sqlite_modern_cpp-3.2",
    urls = ["https://github.com/aminroosta/sqlite_modern_cpp/archive/v3.2.tar.gz"],
)

bind(
    name = "sqlite_modern_cpp",
    actual = "@sqlite_modern_cpp_archive//:sqlite_modern_cpp",
)

git_repository(
    name = "com_github_gflags_gflags",
    commit = "4a694e87361d08eff5c4c9e9f551b1a0d41f7c40",
    remote = "https://github.com/gflags/gflags.git",
)

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

bind(
    name = "gflags_nothreads",
    actual = "@com_github_gflags_gflags//:gflags_nothreads",
)

# boost
http_archive(
    name = "com_github_nelhage_boost",
    strip_prefix = "rules_boost-0838fdac246ef9362b80009b9dd2018b5378a5ed",
    type = "tar.gz",
    urls = [
        "https://github.com/nelhage/rules_boost/archive/0838fdac246ef9362b80009b9dd2018b5378a5ed.tar.gz",
    ],
)

load("@com_github_nelhage_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# blosc
new_http_archive(
    name = "blosc_archive",
    build_file = "third_party/blosc.BUILD",
    urls = [
        "https://github.com/Blosc/c-blosc/archive/v1.12.0.tar.gz",
    ],
)

bind(
    name = "blosc",
    actual = "@blosc_archive//:blosc",
)

# gtest, gmock
new_http_archive(
    name = "gmock_archive",
    build_file = "//third_party:gmock.BUILD",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    strip_prefix = "googletest-release-1.8.0",
    urls = [
        "http://mirror.bazel.build/github.com/google/googletest/archive/release-1.8.0.zip",
        "https://github.com/google/googletest/archive/release-1.8.0.zip",
    ],
)

bind(
    name = "gtest",
    actual = "@gmock_archive//:gtest",
)

bind(
    name = "gtest_main",
    actual = "@gmock_archive//:gtest_main",
)
