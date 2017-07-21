//
// Created by daeyun on 7/15/17.
//

#include "voxel.h"

#include "cpp/lib/common.h"
#include "third_party/src/polygon2voxel/polygon2voxel_double.h"

namespace mvshape {
namespace voxel {

constexpr uint8_t kEmpty = 0;
constexpr uint8_t kFilled = 1;
constexpr uint8_t kVisible = 2;

class VoxelGrid {
 public:
  VoxelGrid(int nx, int ny, int nz) : data_(static_cast<size_t>(nx * ny * nz), 0), nx_(nx), ny_(ny), nz_(nz) {}

  virtual ~VoxelGrid() {}

  inline uint8_t at(int x, int y, int z) const {
    return *(this->data_.data() + x + y * nx_ + z * nx_ * ny_);
  }

  inline uint8_t at(int x) const { return *(this->data_.data() + x); }

  inline void set(uint8_t value, int x, int y, int z) {
    *(this->data_.data() + x + y * nx_ + z * nx_ * ny_) = value;
  }

  inline void reset() { std::fill(data_.begin(), data_.end(), 0); }

  inline void operator|=(const VoxelGrid &other) {
    for (int i = 0; i < this->size(); i++) {
      *(this->data_.data() + i) |= other.at(i);
    }
  }

  int size() const { return nx_ * ny_ * nz_; }

  int nx() const { return nx_; }

  int ny() const { return ny_; }

  int nz() const { return nz_; }

  uint8_t *data() { return data_.data(); }

  const uint8_t *data() const { return data_.data(); }

  void print() {
    for (int x = 0; x < nx_; x++) {
      for (int y = 0; y < ny_; y++) {
        for (int z = 0; z < nz_; z++) {
          auto value = static_cast<int>(this->at(x, y, z));
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

 private:
  vector<uint8_t> data_;
  int nx_, ny_, nz_;
};

void Voxelize(const vector<array<int, 3>> &faces, const vector<array<float, 3>> &vertices,
              const array<int32_t, 3> &volume_nxyz, VoxelGrid *out_volume) {
  Expects(volume_nxyz[0] > 0);
  Expects(volume_nxyz[1] > 0);
  Expects(volume_nxyz[2] > 0);
  Expects(faces.size() > 0);
  Expects(vertices.size() >= 3);

  array<vector<int32_t>, 3> faces_abc;

  faces_abc[0].reserve(faces.size());
  faces_abc[1].reserve(faces.size());
  faces_abc[2].reserve(faces.size());

  for (const auto &face : faces) {
    faces_abc[0].push_back(face[0]);
    faces_abc[1].push_back(face[1]);
    faces_abc[2].push_back(face[2]);
  }

  array<vector<double>, 3> vertices_abc;

  vertices_abc[0].reserve(vertices.size());
  vertices_abc[1].reserve(vertices.size());
  vertices_abc[2].reserve(vertices.size());

  for (const auto &vertex : vertices) {
    vertices_abc[0].push_back(static_cast<double>(vertex[0]));
    vertices_abc[1].push_back(static_cast<double>(vertex[1]));
    vertices_abc[2].push_back(static_cast<double>(vertex[2]));
  }

  *out_volume = VoxelGrid(volume_nxyz[0], volume_nxyz[1], volume_nxyz[2]);

  polygon2voxel(
      faces_abc[0].data(),
      faces_abc[1].data(),
      faces_abc[2].data(),
      static_cast<int32_t>(faces.size()),
      vertices_abc[0].data(),
      vertices_abc[1].data(),
      vertices_abc[2].data(),
      volume_nxyz.data(), 2, out_volume->data());
}

void MarkVisibleXYZ(VoxelGrid *voxels) {
  const int nx = voxels->nx();
  const int ny = voxels->ny();
  const int nz = voxels->nz();

  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
      for (int z = nz - 1; z >= 0; z--) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
    }
  }

  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
      for (int x = nx - 1; x >= 0; x--) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
    }
  }

  for (int z = 0; z < nz; z++) {
    for (int x = 0; x < nx; x++) {
      for (int y = 0; y < ny; y++) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
      for (int y = ny - 1; y >= 0; y--) {
        if (voxels->at(x, y, z) != kFilled) {
          voxels->set(kVisible, x, y, z);
        } else {
          break;
        }
      }
    }
  }
}

}
}
