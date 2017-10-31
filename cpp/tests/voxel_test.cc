//
// Created by daeyun on 6/29/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "cpp/lib/voxel.h"
#include "cpp/lib/resources.h"
#include "cpp/lib/file_io.h"

using namespace mvshape;

TEST(Voxelize, Mesh) {
  // From ShapenetCore.v1 class 03001627.
  auto filename = Resources::ResourcePath("objects/1a74a83fa6d24b3cacd67ce2c72c02e/model.obj");
  std::vector<std::array<int, 3>> faces;
  std::vector<std::array<float, 3>> vertices;
  FileIO::ReadFacesAndVertices(filename, &faces, &vertices);

  int res = 100;

  auto voxel_grid = voxel::VoxelGrid(res, res, res);

  for (int i = 0; i < voxel_grid.size(); ++i) {
    EXPECT_EQ(0, voxel_grid.at(i));
  }

  float size = 1.0;
  voxel::Voxelize(faces, vertices, res, 0, 1.0, {-size, -size, -size}, {size, size, size}, &voxel_grid);

  int ones = 0;
  for (int i = 0; i < voxel_grid.size(); ++i) {
    EXPECT_TRUE(voxel_grid.at(i) == 0 or voxel_grid.at(i) == 1);
    ones += static_cast<int>(voxel_grid.at(i));
  }
  EXPECT_LT(150, ones);
}

TEST(Voxel, Copy) {
  auto grid = voxel::VoxelGrid(10, 10, 10);
  EXPECT_EQ(0, grid.at(0));

  grid.set(1, 0);
  EXPECT_EQ(1, grid.at(0));

  auto grid2 = grid;
  EXPECT_EQ(1, grid2.at(0));

  grid.set(2, 0);
  EXPECT_EQ(2, grid.at(0));
  EXPECT_EQ(1, grid2.at(0));
}

TEST(Voxel, FillEmpty) {
  auto voxel_grid = voxel::VoxelGrid(32, 32, 32);

  for (int j = 0; j < voxel_grid.size(); ++j) {
    const uint8_t value = voxel_grid.at(j);
    if (value == voxel::kEmpty) {
      voxel_grid.set(voxel::kUnknown, j);
    }
  }

  for (int j = 0; j < voxel_grid.size(); ++j) {
    const uint8_t value = voxel_grid.at(j);
    EXPECT_EQ(voxel::kUnknown, value);
  }

  VoxFill(&voxel_grid, 13);

  for (int j = 0; j < voxel_grid.size(); ++j) {
    const uint8_t value = voxel_grid.at(j);
    EXPECT_EQ(voxel::kVisible, value);
  }
}
