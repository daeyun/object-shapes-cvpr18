//
// Created by daeyun on 6/7/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <gtest/gtest.h>

#include "file_io.h"
#include "resources.h"
#include "random_utils.h"
#include "single_batch_loader.h"

using namespace mvshape;

TEST(SingleBatchLoader, Read) {
  auto dirname = Resources::ResourcePath("data/shrec12/mv6_depth_128");
  auto files = FileIO::RegularFilesInDirectory(dirname);

  vector<int> shape{static_cast<int>(files.size()), 6, 128, 128, 1};
  vector<float> data;
  data.resize(files.size() * 6 * 128 * 128);

  EXPECT_EQ(0.0, data[0]);
  ReadSingleBatch(files, shape, data.data());
  EXPECT_NE(0.0, data[0]);
  EXPECT_TRUE(std::isnan(data[0]));
}

TEST(SingleBatchLoader, PreserveOrder) {
  auto dirname = Resources::ResourcePath("data/shrec12/mv6_depth_128");
  auto files = FileIO::RegularFilesInDirectory(dirname);

  for (int j = 0; j < 3; ++j) {
    Random::Shuffle(files.begin(), files.end());

    vector<int> shape{static_cast<int>(files.size()), 6, 128, 128, 1};
    vector<float> data;
    data.resize(files.size() * 6 * 128 * 128);
    ReadSingleBatch(files, shape, data.data());

    vector<int> shape2;
    vector<float> data2;

    FileIO::ReadTensorData(files[0], &shape2, &data2);
    EXPECT_EQ(shape2[0], 6);
    EXPECT_EQ(shape2[1], 128);
    EXPECT_EQ(shape2[2], 128);
    EXPECT_EQ(0, memcmp(data.data(), data2.data(), 6 * 128 * 128));

    // skip exponentially
    for (int i = 1; i < files.size(); i *= 2) {
      shape2.clear();
      data2.clear();
      FileIO::ReadTensorData(files[i], &shape2, &data2);
      EXPECT_EQ(0, memcmp(data.data() + i * 6 * 128 * 128, data2.data(), 6 * 128 * 128));
    }
  }

}

