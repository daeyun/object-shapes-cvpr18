//
// Created by daeyun on 6/20/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gflags/gflags.h>
#include <glm/glm.hpp>
#include <boost/filesystem.hpp>

#include "glog/logging.h"

#include "cpp/lib/transforms.h"
#include "cpp/lib/data_io.h"
#include "cpp/lib/file_io.h"
#include "cpp/lib/string_utils.h"
#include "cpp/lib/resources.h"
#include "cpp/lib/transforms.h"
#include "cpp/lib/database.h"
#include "cpp/lib/flags.h"
#include "cpp/lib/egl_rendering.h"  // Should be the last include.

using namespace mvshape;
namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  Data::GenerateDataset();
}
