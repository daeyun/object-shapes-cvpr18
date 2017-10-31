#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>

#include "cpp/lib/data_io.h"
#include "cpp/lib/database.h"
#include "cpp/lib/flags.h"

using namespace mvshape;
namespace mv = mvshape_dataset;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

//  mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_vpo",
//                              "splits/shrec12_examples_vpo/");
//
//  mvshape::Data::SaveExamples("database/shrec12.sqlite", "shrec12_examples_opo",
//                              "splits/shrec12_examples_opo/");

  mvshape::Data::SaveExamples("database/shapenetcore.sqlite", "shapenetcore_examples_vpo",
                              "splits/shapenetcore_examples_vpo/");

  mvshape::Data::SaveExamples("database/shapenetcore.sqlite", "shapenetcore_examples_opo",
                              "splits/shapenetcore_examples_opo/");
}
