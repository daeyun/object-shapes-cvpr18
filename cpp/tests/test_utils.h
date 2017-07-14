//
// Created by daeyun on 7/13/17.
//

#pragma once

#include <gtest/gtest.h>

#include "common.h"
#include "flags.h"
#include "file_io.h"
#include "random_utils.h"

class OutputTest : public ::testing::Test {
 protected:
  OutputTest() : out_dir(FLAGS_out_dir) {
    FLAGS_out_dir = mvshape::FileIO::JoinPath(FLAGS_out_dir, mvshape::Random::RandomString(8));
  }

  virtual ~OutputTest() {
    // Parent directory will be deleted in main().
    FLAGS_out_dir = out_dir;
  }

  std::string out_dir;
};

