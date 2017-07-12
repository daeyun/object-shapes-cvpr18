//
// Created by daeyun on 6/20/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <gtest/gtest.h>

#include "common.h"
#include "string_utils.h"

using namespace mvshape;

TEST(Split, SplitFloats) {
  auto values = SplitValues<float>("1,2, 3");

  EXPECT_EQ(values.size(), 3);
  EXPECT_EQ(values[0], 1.0f);
  EXPECT_EQ(values[1], 2.0f);
  EXPECT_EQ(values[2], 3.0f);
}

TEST(Split, SplitInvalidFloats) {
  ASSERT_THROW(SplitValues<float>("1,2,a,3"), std::runtime_error);
}

TEST(Split, ParseVec3) {
  Vec3 vec = ParseMatrix<3, 1>("1, 2,3");
  EXPECT_EQ(vec[0], 1.0);
  EXPECT_EQ(vec[1], 2.0);
  EXPECT_EQ(vec[2], 3.0);
}

TEST(Split, ParseVec3Invalid) {
  ASSERT_NO_THROW(ParseMatrix<3>("1,2,3"));
  ASSERT_ANY_THROW(ParseMatrix<3>("1,2"));
}
