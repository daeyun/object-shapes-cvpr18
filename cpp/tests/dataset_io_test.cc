//
// Created by daeyun on 6/29/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <unordered_map>

#include <gsl/gsl_assert>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "flags.h"
#include "proto/dataset.pb.h"
#include "database.h"
#include "resources.h"
#include "data_io.h"

using namespace mvshape;
namespace mv = mvshape_dataset;

TEST(GenerateDataset, Subset) {
  Data::GenerateDataset();
}

TEST(GenerateMetadata, Subset) {
  int count = mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_vpo",
                                          "splits/shrec12_examples_vpo/");
  EXPECT_EQ(20, count);
}

TEST(LoadMetadata, Subset) {
  mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_vpo",
                              "splits/shrec12_examples_vpo/");

  mvshape_dataset::Examples examples;
  int train = mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &examples);
  EXPECT_EQ(2, train);
  int test = mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);
  EXPECT_EQ(6, test);

  EXPECT_TRUE(!examples.examples(0).single_depth().filename().empty());

  EXPECT_EQ(examples.examples(0).single_depth().object_id(),
            examples.examples(0).multiview_depth().object_id());

  for (auto &example: examples.examples()) {
    LOG(INFO) << example.single_depth().filename();
  }
}

TEST(LoadMetadata, MatchInputOutput) {
  mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_vpo",
                              "splits/shrec12_examples_vpo/");

  mvshape_dataset::Examples examples;
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/validation.bin"), &examples);
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &examples);

  for (int i = 0; i < examples.examples_size(); ++i) {
    EXPECT_EQ(examples.examples(i).single_depth().object_id(), examples.examples(i).multiview_depth().object_id());
    EXPECT_NEAR(examples.examples(i).single_depth().eye()[0], examples.examples(i).multiview_depth().eye()[0], 1e-5);
  }
}

class ReadExamplesTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    Data::GenerateDataset();
    mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_vpo",
                                "splits/shrec12_examples_vpo/");
  }
};

TEST_F(ReadExamplesTest, ValidSize) {
  mvshape_dataset::Examples examples;
  int test = mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);
  EXPECT_EQ(6, test);

  string datadir = FLAGS_data_dir;
  FLAGS_data_dir = FLAGS_out_dir;
  string data;
  const auto &depth_rendering = examples.examples(0).single_depth();
  Data::LoadRenderingContent(depth_rendering, &data);

  // 65536
  auto expected = 4 * depth_rendering.resolution() * depth_rendering.resolution() * depth_rendering.set_size();
  EXPECT_EQ(expected, data.size());

  const auto &normal_rendering = examples.examples(0).multiview_depth();
  Data::LoadRenderingContent(normal_rendering, &data);

  // 65536 * v
  expected = 4 * normal_rendering.resolution() * normal_rendering.resolution() * normal_rendering.set_size();
  EXPECT_EQ(expected, data.size());

  FLAGS_data_dir = datadir;
}

TEST_F(ReadExamplesTest, ExampleReader) {
  mvshape_dataset::Examples examples;
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);
  auto reader = Data::RenderingReader(&examples);
  string datadir = FLAGS_data_dir;
  FLAGS_data_dir = FLAGS_out_dir;

  std::unordered_map<int, vector<unique_ptr<string>>> data;
  reader.LoadBatch({
                       mv::Example::kSingleDepthFieldNumber,
                       mv::Example::kMultiviewDepthFieldNumber,
                   }, 3, &data);

  EXPECT_EQ(2, data.size());
  EXPECT_EQ(3, data[mv::Example::kSingleDepthFieldNumber].size());
  EXPECT_EQ(65536, data[mv::Example::kSingleDepthFieldNumber][0]->size());
  EXPECT_EQ(6 * 65536, data[mv::Example::kMultiviewDepthFieldNumber][0]->size());
  FLAGS_data_dir = datadir;
}

TEST_F(ReadExamplesTest, Queues) {
  mvshape_dataset::Examples examples;
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);

  int batch_size = 100;

  string datadir = FLAGS_data_dir;
  FLAGS_data_dir = FLAGS_out_dir;

  mvshape::Data::BatchLoader loader(&examples, {
      mv::Example::kSingleDepthFieldNumber,
      mv::Example::kMultiviewDepthFieldNumber,
  }, batch_size, true, true);

  loader.StartWorkers();

  for (int i = 0; i < 10; ++i) {
    auto data = loader.Next();
    EXPECT_EQ(2, data->file_fields.size());
  }

  loader.StopWorkers();

  loader.StartWorkers();

  for (int i = 0; i < 5; ++i) {
    auto data = loader.Next();
    EXPECT_EQ(2, data->file_fields.size());
  }

  loader.StopWorkers();

  FLAGS_data_dir = datadir;
}

TEST_F(ReadExamplesTest, MatchInputOutput) {
  mvshape_dataset::Examples examples;
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &examples);
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/validation.bin"), &examples);
  mvshape::Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &examples);

  EXPECT_EQ(14, examples.examples_size());

  int batch_size = 1;
  LOG(INFO) << "batch_size " << batch_size;

  string datadir = FLAGS_data_dir;
  FLAGS_data_dir = FLAGS_out_dir;

  mvshape::Data::BatchLoader loader(&examples, {
      mv::Example::kSingleDepthFieldNumber,
      mv::Example::kMultiviewDepthFieldNumber,
  }, batch_size, true, false);

  loader.StartWorkers();

  std::hash<std::string> str_hash;
  std::map<size_t, size_t> mapping;

  for (int i = 0; i < 2000; ++i) {
    auto data = loader.Next();
    for (int j = 0; j < data->file_fields.size(); ++j) {
      data->file_fields[mv::Example::kSingleDepthFieldNumber];
      size_t hash1 = str_hash(data->file_fields[mv::Example::kSingleDepthFieldNumber]);
      size_t hash2 = str_hash(data->file_fields[mv::Example::kMultiviewDepthFieldNumber]);
      if (mapping.find(hash1) == mapping.end()) {
        mapping[hash1] = hash2;
      } else {
        ASSERT_EQ(mapping[hash1], hash2);
      }
    }
  }

  loader.StopWorkers();

  FLAGS_data_dir = datadir;
}
