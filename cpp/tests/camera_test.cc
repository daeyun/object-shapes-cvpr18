//
// Created by daeyun on 6/29/17.
//

#include <gtest/gtest.h>

#include "camera.h"

using namespace mvshape;

TEST(PerspectiveCamera, gluPerspective) {
  auto frustum = FrustumParams::MakePerspective(20, 1, 0.1, 10);
  auto cam = PerspectiveCamera({0, 0, 1}, {0, 0, 0}, {0, 1, 0}, frustum);
  auto mat = cam.projection_mat();
  EXPECT_NEAR(5.6713, mat(0, 0), 1e-3);
  EXPECT_NEAR(5.6713, mat(1, 1), 1e-3);
  EXPECT_NEAR(-1.0202, mat(2, 2), 1e-3);
  EXPECT_NEAR(-1, mat(3, 2), 1e-5);
  EXPECT_NEAR(-0.202, mat(2, 3), 1e-3);
}
