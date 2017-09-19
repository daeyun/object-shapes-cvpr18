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
#include "cpp/lib/random_utils.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <cpp/lib/file_io.h>
#include <cpp/lib/fssr_utils.h>

using namespace mvshape;
namespace mv = mvshape_dataset;
using boost::property_tree::ptree;
using boost::filesystem::path;

DEFINE_string(mesh1, "", "GT mesh");
DEFINE_string(mesh2, "", "recon mesh");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  auto triangles1 = FileIO::ReadTriangles(FLAGS_mesh1);
  auto triangles2 = FileIO::ReadTriangles(FLAGS_mesh2);

  std::vector<meshdist_cgal::Triangle> tri1, tri2;

  for (const auto &item : triangles1) {
    tri1.push_back(meshdist_cgal::Triangle(
        Vec3{item[0][0], item[0][1], item[0][2]},
        Vec3{item[1][0], item[1][1], item[1][2]},
        Vec3{item[2][0], item[2][1], item[2][2]}
    ));
  }
  for (const auto &item : triangles2) {
    tri2.push_back(meshdist_cgal::Triangle(
        Vec3{item[0][0], item[0][1], item[0][2]},
        Vec3{item[1][0], item[1][1], item[1][2]},
        Vec3{item[2][0], item[2][1], item[2][2]}
    ));
  }

  auto mean_rms = meshdist_cgal::MeshToMeshDistance(tri1, tri2);

  // Points are shuffled.
  Points3d points1, points2;
  meshdist_cgal::SamplePointsOnTriangles(tri1, 900, &points1);
  meshdist_cgal::SamplePointsOnTriangles(tri1, 900, &points2);

  auto mean = (points1 - points2).colwise().norm().mean();
  LOG(INFO) << "MEAN RMS: " << mean_rms / mean;
}
