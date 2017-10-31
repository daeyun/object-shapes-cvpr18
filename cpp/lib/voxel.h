//
// Created by daeyun on 7/15/17.
//

#pragma once

#include "cpp/lib/common.h"

namespace mvshape {
namespace voxel {

constexpr uint8_t kEmpty = 0;
constexpr uint8_t kFilled = 1;
constexpr uint8_t kVisible = 2;
constexpr uint8_t kUnknown = 3;

class VoxelGrid {
 public:
  VoxelGrid(int nx, int ny, int nz) : data_(static_cast<size_t>(nx * ny * nz), kEmpty), nx_(nx), ny_(ny), nz_(nz) {}

  virtual ~VoxelGrid() = default;

  inline uint8_t at(int x, int y, int z) const {
    return *(this->data_.data() + x + y * nx_ + z * nx_ * ny_);
  }

  inline uint8_t at(int i) const { return *(this->data_.data() + i); }

  inline void set(uint8_t value, int x, int y, int z) {
    *(this->data_.data() + x + y * nx_ + z * nx_ * ny_) = value;
  }

  inline void set(uint8_t value, int i) {
    *(this->data_.data() + i) = value;
  }

  inline void reset() { std::fill(data_.begin(), data_.end(), kEmpty); }

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

  void save(const std::string &filename);

 private:
  vector<uint8_t> data_;
  int nx_, ny_, nz_;
};

void Voxelize(const vector<array<int, 3>> &faces,
              const vector<array<float, 3>> &vertices,
              int resolution,
              int padding,
              float scale,
              std::array<float, 3> bmin,
              std::array<float, 3> bmax,
              VoxelGrid *out_volume);

void SweepXYZAndMarkVisible(VoxelGrid *voxels);
bool RayBackTracing(const array<int, 3> &starting_pos, const array<float, 3> &direction,
                    const VoxelGrid &voxels, VoxelGrid *directional_output);
int VoxFill(VoxelGrid* voxels, int window_size);

}
}

