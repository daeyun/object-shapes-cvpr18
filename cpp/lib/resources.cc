//
// Created by daeyun on 6/14/17.
//

#include "resources.h"

#include <fstream>

#include <gflags/gflags.h>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

#include "flags.h"

namespace mvshape {
namespace Resources {

// Platform dependent.
std::string PathToExecutable() {
  char buf[1024] = {0};
  ssize_t size = readlink("/proc/self/exe", buf, sizeof(buf));
  if (size == 0 || size == sizeof(buf)) {
    // Fallback to working directory.
    return boost::filesystem::current_path().string();
  }
  std::string path(buf, size);
  boost::system::error_code ec;
  boost::filesystem::path p(
      boost::filesystem::canonical(
          path, boost::filesystem::current_path(), ec));
  return p.make_preferred().string();
}

std::string FindResourceDir() {
  static string resource_path = "";
  if (!resource_path.empty()) {
    return resource_path;
  }

  boost::filesystem::path p;
  if (!FLAGS_resources_dir.empty()) {
    p = boost::filesystem::path(FLAGS_resources_dir);
  } else {
    auto path = boost::filesystem::path(PathToExecutable());
    while (true) {
      if (path.string() == "/") {
        LOG(FATAL) << "Could not find resources directory.";
        p = "./resources";
        break;
      }
      if (boost::filesystem::is_directory(path / "resources")) {
        p = path / "resources";
        break;
      }
      path = path.parent_path();
    }
  }

  // Preserves symlinks.
  auto ret = boost::filesystem::absolute(p).string();
  resource_path = ret;
  LOG(INFO) << "Path to 'resources' dir: " << resource_path;
  return ret;
}

std::string ResourcePath(const std::string &filename) {
  return (boost::filesystem::path(FindResourceDir()) / filename).string();
}

std::string ReadResource(const std::string &filename) {
  return FileIO::ReadBytes(ResourcePath(filename));
}
}  // Resources
}  // mvshape
