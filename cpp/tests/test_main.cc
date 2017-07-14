#include <stdio.h>

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flags.h"
#include "file_io.h"
#include "resources.h"
#include "random_utils.h"

DEFINE_bool(remove_out_dir, true, "Clean up the temp output directory.");

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from test_main.cc\n\n");
  fflush(stdout);
  fflush(stderr);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_is_test_mode = 1;

  FLAGS_resources_dir = mvshape::Resources::FindResourceDir();
  FLAGS_data_dir = FLAGS_resources_dir + "/data";
  FLAGS_out_dir = mvshape::FileIO::NamedEmptyTempDir("mvshape_test_" + mvshape::Random::RandomString(8));
  testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  if (FLAGS_remove_out_dir) {
    mvshape::FileIO::RemoveDirIfExists(FLAGS_out_dir);
  }

  return ret;
}
