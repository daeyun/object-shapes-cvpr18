#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>
#include <third_party/repos/mve/libs/mve/view.h>
#include <third_party/repos/mve/libs/util/file_system.h>

#include "cpp/lib/data_io.h"
#include "cpp/lib/database.h"
#include "cpp/lib/flags.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <cpp/lib/file_io.h>
#include <cpp/lib/fssr_utils.h>

using namespace mvshape;
namespace mv = mvshape_dataset;
using boost::property_tree::ptree;
using boost::filesystem::path;

DEFINE_string(tensor_dir, "", "e.g. tf_out/depth_6_views/object_centered/tensors/0040_000011400/NOVELVIEW");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  vector<string> tensor_dirs;
  auto tensor_dir = FileIO::FullOutPath(FLAGS_tensor_dir);
  if (FileIO::Exists((path(tensor_dir) / "index/").string())) {
    tensor_dirs.push_back(tensor_dir);
  } else {
    auto dirs = FileIO::DirectoriesInDirectory(tensor_dir);
    for (const auto &dir : dirs) {
      if (FileIO::Exists((path(dir) / "index/").string())) {
        tensor_dirs.push_back(dir);
      }
    }
  }

  Ensures(tensor_dirs.size() > 0);

  for (const auto &input_dir : tensor_dirs) {
    auto dirname = path(input_dir).remove_trailing_separator().parent_path().parent_path().parent_path();

    auto mesh_out_dir =
        dirname / "recon"
            / path(input_dir).remove_trailing_separator().parent_path().stem()
            / path(input_dir).remove_trailing_separator().stem();

    mvshape::fssr::Depth2Mesh(input_dir, mesh_out_dir.string());
  }

}
