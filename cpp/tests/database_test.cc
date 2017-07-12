//
// Created by daeyun on 6/20/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "database.h"
#include "resources.h"
#include "flags.h"
#include "file_io.h"
#include "proto/dataset.pb.h"

using namespace mvshape;
namespace mv = mvshape_dataset;

TEST(Database, ReadRenderables) {
  vector<mv::Rendering> data;
  // FLAGS_data_dir is set to resources directory in test_main.cc.
  int count = mvshape::Data::ReadRenderables("database/shrec12.sqlite", "shrec12_renderings", &data);
  EXPECT_EQ(120, count);
  for (const auto &item : data) {
    EXPECT_EQ(3, item.eye_size());
    EXPECT_EQ(3, item.up_size());
    EXPECT_EQ(3, item.lookat_size());
    for (int i = 0; i < 3; ++i) {
      EXPECT_LT(-20, item.eye(i));
      EXPECT_GT(20, item.eye(i));
    }
    EXPECT_NEAR(1, Vec3(item.up(0), item.up(1), item.up(2)).stableNorm(), 1e-5);
    EXPECT_TRUE(FileIO::Exists(FileIO::FullDataPath(item.mesh_filename())));
  }
}

TEST(Database, ReadExamples) {
  mv::Examples aa;
  aa.set_split_name(mv::TEST);
  aa.set_dataset_name(mv::SHREC12);
  auto bb = aa.add_examples();
  bb->set_id(32);

  std::map<mv::Split, mv::Examples> data;
  // FLAGS_data_dir is set to "resources" directory in test_main.cc.
  int count = mvshape::Data::ReadExamples("database/shrec12.sqlite", "shrec12_examples_vpo", &data);
  EXPECT_EQ(4, data.size());
  EXPECT_EQ(6, data[mv::TEST].examples_size());
  EXPECT_EQ(mv::TRAIN, data[mv::TRAIN].split_name());
  EXPECT_EQ(mv::SHREC12, data[mv::TRAIN].dataset_name());
  EXPECT_LT(0, data[mv::TEST].examples(0).single_depth().filename().size());
  EXPECT_LT(1000, data[mv::TRAIN].SerializeAsString().size());
}

TEST(Database, VerifyProtobufVersion) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}
