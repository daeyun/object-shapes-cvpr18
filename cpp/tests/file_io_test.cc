//
// Created by daeyun on 6/7/17.
//

#include <gtest/gtest.h>

#include "file_io.h"
#include "resources.h"

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