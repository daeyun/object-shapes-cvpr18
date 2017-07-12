//
// Created by daeyun on 6/30/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "multiprocessing.h"

using namespace mvshape;

TEST(SplitIntoNChunks, TryAll) {
  for (int size = 0; size < 200; ++size) {
    for (int n = 1; n < 400; ++n) {
      auto ranges = MP::SplitRange(size, n);

      int count = 0;
      int maxsize = INT_MIN;
      int minsize = INT_MAX;
      for (const auto &range : ranges) {
        maxsize = std::max(maxsize, range.second - range.first);
        minsize = std::min(minsize, range.second - range.first);
        for (int i = range.first; i < range.second; ++i) {
          EXPECT_EQ(count, i);
          ++count;
        }
      }
      EXPECT_EQ(count, size);
      EXPECT_GE(1, maxsize - minsize);
    }
  }
}
