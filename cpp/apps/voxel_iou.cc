//
// Created by daeyun on 11/15/17.
//


#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <iostream>

#include "cpp/lib/voxel.h"
#include "cpp/lib/file_io.h"
#include "cpp/lib/data_io.h"

using namespace mvshape;

int main(int argc, char *argv[]) {

  auto compute_iou = [](const std::string &filename1, const std::string &filename2) {
    auto read_voxels = [](const std::string &fname) {
      std::vector<std::array<int, 3>> faces;
      std::vector<std::array<float, 3>> vertices;

      int resolution = 32;
      int padding = 0;
      float scale = 1.0;
      // unit cube at origin
      std::array<float, 3> bmin = {-0.5, -0.5, -0.5};
      std::array<float, 3> bmax = {0.5, 0.5, 0.5};
      voxel::VoxelGrid out_volume(resolution, resolution, resolution);

      bool ok = FileIO::ReadFacesAndVertices(fname, &faces, &vertices);
      if (!ok) {
        return out_volume;
      }

      voxel::Voxelize(faces, vertices, resolution, padding, scale, bmin, bmax, &out_volume);
      return out_volume;
    };

    auto v1 = read_voxels(filename1);
    auto v2 = read_voxels(filename2);

    v1.save(filename1 + ".vox.bin");
    v2.save(filename2 + ".vox.bin");

    float iou = v1.iou(v2);
    return iou;
  };

//  std::string filename1 = "/data/mvshape/out/pascal3d_recon/o/aeroplane_01_n02690373_1277/gt.off";
//  std::string filename2 = "/data/mvshape/out/pascal3d_recon/o/bus_01_n02924116_32040/gt.off";

  auto dirnames = FileIO::DirectoriesInDirectory("/data/mvshape/out/pascal3d_recon/o/");

  for (const auto &item : dirnames) {
    std::string out_filename = item + "/iou_recon.txt";
    if (FileIO::Exists(out_filename)) {
      std::cout << "Skipping " << item << std::endl;
      continue;
    }

    std::string filename1 = item + "/gt.off";
    std::string filename2 = item + "/recon/fssr_recon.ply";
    std::cout << item << std::endl;
    float iou = compute_iou(filename1, filename2);
    std::cout << iou << std::endl;

    std::ofstream myfile;
    myfile.open(out_filename);
    myfile << iou;
    myfile.close();
  }

}
