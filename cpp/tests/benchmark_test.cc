//
// Created by daeyun on 6/29/17.
//

#include <gtest/gtest.h>
#include <thread>
#include <chrono>

#include "benchmark.h"

using namespace mvshape;

TEST(BenchmarkUtils, Epoch) {
  auto time = MicroSecondsSinceEpoch();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  auto elapsed_ms = static_cast<double>(MicroSecondsSinceEpoch() - time) / 1000;
  EXPECT_NEAR(elapsed_ms, 500, 10);
}
