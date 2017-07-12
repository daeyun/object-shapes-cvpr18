//
// Created by daeyun on 6/29/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "common.h"
#include "random_utils.h"

using namespace mvshape;

TEST(Random, ShuffleEmpty) {
  vector<int> values{};
  mvshape::Random::Shuffle(values.begin(), values.end());
  EXPECT_TRUE(values.empty());
}

TEST(Random, ShuffleOne) {
  vector<int> values{42};
  mvshape::Random::Shuffle(values.begin(), values.end());
  EXPECT_EQ(1, values.size());
  EXPECT_EQ(42, values[0]);
}

TEST(Random, Shuffle) {
  vector<int> values{0, 1, 2, 3, 4};
  mvshape::Random::Shuffle(values.begin(), values.end());
  EXPECT_EQ(5, values.size());
}

TEST(Random, Choose0) {
  vector<int> values{0, 1, 2, 3, 4};
  vector<int> chosen;
  mvshape::Random::ChooseN<int>(0, &values, &chosen);
  EXPECT_EQ(0, chosen.size());
}

TEST(Random, ChooseN) {
  vector<int> values{0, 1, 2, 3, 4, 42};

  for (int i = 0; i < 6; ++i) {
    vector<int> chosen;
    mvshape::Random::ChooseN<int>(i, &values, &chosen);
    EXPECT_EQ(i, chosen.size());

    std::sort(chosen.begin(), chosen.end());
    EXPECT_TRUE(std::unique(chosen.begin(), chosen.end()) == chosen.end());
  }

  vector<int> chosen;
  mvshape::Random::ChooseN<int>(7, &values, &chosen);
  EXPECT_EQ(7, chosen.size());

  std::sort(chosen.begin(), chosen.end());
  EXPECT_FALSE(std::unique(chosen.begin(), chosen.end()) == chosen.end());
}
