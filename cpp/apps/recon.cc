//
// Created by daeyun on 9/21/17.
//

#include <fstream>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <sys/wait.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <lrucache.hpp>

#include <iomanip>

#include "common.h"
#include "database.h"
#include "file_io.h"
#include "transforms.h"
#include "egl_rendering.h"
#include "benchmark.h"
#include "multiprocessing.h"
#include "random_utils.h"
#include "proto/dataset.pb.h"

namespace fs = boost::filesystem;
namespace mv = mvshape_dataset;

using namespace mvshape;


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  std::string filename = FileIO::FullDataPath("/mesh/shrec12/TestSubset/NovelClass/ApartmentHouse_aligned/D00045_out.off");
  vector<array<array<float, 3>, 3>> triangles = FileIO::ReadTriangles(filename);

  int resolution = 128;
  Rendering::RendererConfig render_config{
      .width = resolution,
      .height = resolution,
  };

  auto renderer = Rendering::ShapeRenderer(render_config);

  renderer.Render();
}
