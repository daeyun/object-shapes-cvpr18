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
#include "cpp/lib/meshdist.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <cpp/lib/file_io.h>
#include <cpp/lib/fssr_utils.h>

using namespace mvshape;
namespace mv = mvshape_dataset;
using boost::property_tree::ptree;
using boost::filesystem::path;

DEFINE_string(mesh1, "", "");
DEFINE_string(mesh2, "", "");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  auto triangles1 = FileIO::ReadTriangles(FLAGS_mesh1);
  auto triangles2 = FileIO::ReadTriangles(FLAGS_mesh2);

  std::vector<meshdist::Triangle> tri1, tri2;

  for (const auto &item : triangles1) {
    tri1.push_back(meshdist::Triangle(
        meshdist::Vec3{item[0][0], item[0][1], item[0][2]},
        meshdist::Vec3{item[1][0], item[1][1], item[1][2]},
        meshdist::Vec3{item[2][0], item[2][1], item[2][2]}
    ));
  }
  for (const auto &item : triangles2) {
    tri2.push_back(meshdist::Triangle(
        meshdist::Vec3{item[0][0], item[0][1], item[0][2]},
        meshdist::Vec3{item[1][0], item[1][1], item[1][2]},
        meshdist::Vec3{item[2][0], item[2][1], item[2][2]}
    ));
  }

  meshdist::MeshToMeshDistance(tri1, tri2);
}
