//
// Created by daeyun on 6/7/17.
//

#include <gtest/gtest.h>
#include <glog/logging.h>

#include "file_io.h"
#include "resources.h"
#include "boost/filesystem/path.hpp"

using namespace mvshape;

TEST(FileIO, TestCompressionIdentity) {
  std::vector<float> data;
  for (int i = 0; i < 50; ++i) {
    for (int j = 0; j < 50; ++j) {
      data.push_back(static_cast<float>(i));
    }
  }

  auto data_ptr = reinterpret_cast<const char *>(data.data());
  auto size = data.size() * sizeof(int);

  std::string compressed;
  mvshape::FileIO::CompressBytes(data_ptr, size, "lz4", 9, sizeof(float), &compressed);

  EXPECT_LT(compressed.size(), size);

  std::string decompressed;
  mvshape::FileIO::DecompressBytes(compressed.data(), &decompressed);

  ASSERT_EQ(size, decompressed.size());
  EXPECT_TRUE(std::equal(data_ptr, data_ptr + size, decompressed.data()));
}

TEST(FileIO, ReadTriangleMesh) {
  // From ShapenetCore.v1 class 03001627.
  auto filename = Resources::ResourcePath("objects/1a74a83fa6d24b3cacd67ce2c72c02e/model.obj");
  auto triangles = FileIO::ReadTriangles(filename);
  EXPECT_EQ(triangles.size(), 8376);
}

TEST(FileIO, ListFilesInDir) {
  auto dirname = Resources::ResourcePath("objects/1a74a83fa6d24b3cacd67ce2c72c02e/");
  auto files = FileIO::RegularFilesInDirectory(dirname);
  EXPECT_EQ(2, files.size());
}

TEST(FileIO, LastPathComponent) {
  EXPECT_EQ("3", boost::filesystem::path("/1/2/3").remove_trailing_separator().stem().string());

  std::string s = "/1/2/3/";
  EXPECT_EQ("3", boost::filesystem::path(s).remove_trailing_separator().stem().string());

  EXPECT_EQ("/1/2/3/", s);
}

TEST(FileIO, ReadTensor) {
  auto filename = Resources::ResourcePath("tf_out/0040_000011400/NOVELCLASS/placeholder_target_depth/0.bin");
  vector<int> shape;
  vector<float> data;
  FileIO::ReadTensorData(filename, &shape, &data);

  EXPECT_EQ(5, shape.size());

  EXPECT_EQ(5, shape[0]);
  EXPECT_EQ(6, shape[1]);
  EXPECT_EQ(128, shape[2]);
  EXPECT_EQ(128, shape[3]);
  EXPECT_EQ(1, shape[4]);

  EXPECT_TRUE(std::isnan(data[0]));
  EXPECT_NEAR(5.7774992, data[40000], 1e-5);
}