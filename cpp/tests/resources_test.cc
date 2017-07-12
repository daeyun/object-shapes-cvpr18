//
// Created by daeyun on 6/14/17.
//

#include <gtest/gtest.h>

#include "resources.h"

using namespace mvshape;

TEST(Resources, ReadFile) {
  std::string content = Resources::ReadResource("objects/1a74a83fa6d24b3cacd67ce2c72c02e/model.obj");
  EXPECT_LT(1000, content.size());
}
