//
// Created by daeyun on 7/15/17.
//

#include "voxel.h"
#include "file_io.h"

#include "third_party/src/polygon2voxel/polygon2voxel_double.h"

namespace mvshape {
namespace voxel {

constexpr int kRayWindowSize = 13;

void Voxelize(const vector<array<int, 3>> &faces,
              const vector<array<float, 3>> &vertices,
              int resolution,
              int padding,
              float scale,
              std::array<float, 3> bmin,
              std::array<float, 3> bmax,
              VoxelGrid *out_volume) {
  Expects(resolution > 0);
  Expects(padding >= 0);
  Expects(faces.size() > 0);
  Expects(vertices.size() >= 3);
  Expects(bmin[0] < bmax[0]);
  Expects(bmin[1] < bmax[1]);
  Expects(bmin[2] < bmax[1]);
  if (scale <= 0) {
    scale = 1.0;
  }

  vector<array<float, 3>> new_vertices = vertices;

  float s = std::fmax(std::fmax(bmax[0] - bmin[0], bmax[1] - bmin[1]), bmax[2] - bmin[2]);
  for (auto &vertex : new_vertices) {
    for (int i = 0; i < 3; ++i) {
      vertex[i] *= scale;
      vertex[i] -= bmin[i];
      vertex[i] *= static_cast<float>(resolution - 1 - padding * 2) / s;
      vertex[i] += padding;
    }
  }

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

  vertices_abc[0].reserve(new_vertices.size());
  vertices_abc[1].reserve(new_vertices.size());
  vertices_abc[2].reserve(new_vertices.size());

  for (const auto &vertex : new_vertices) {
    vertices_abc[0].push_back(static_cast<double>(vertex[0]));
    vertices_abc[1].push_back(static_cast<double>(vertex[1]));
    vertices_abc[2].push_back(static_cast<double>(vertex[2]));
  }

  auto volume = VoxelGrid(resolution, resolution, resolution);

  const array<int32_t, 3> volume_nxyz{resolution, resolution, resolution};

  polygon2voxel(faces_abc[0].data(),
                faces_abc[1].data(),
                faces_abc[2].data(),
                static_cast<int32_t>(faces.size()),
                vertices_abc[0].data(),
                vertices_abc[1].data(),
                vertices_abc[2].data(),
                volume_nxyz.data(), 2, volume.data());

  // Transpose. Swap x and z.
  *out_volume = VoxelGrid(volume.nz(), volume.ny(), volume.nx());
  for (int z = 0; z < volume.nz(); ++z) {
    for (int y = 0; y < volume.ny(); ++y) {
      for (int x = 0; x < volume.nx(); ++x) {
        out_volume->set(volume.at(x, y, z), z, y, x);
      }
    }
  }

  // Initially we don't know which voxels are actually empty. Set to unknown.
  for (int j = 0; j < out_volume->size(); ++j) {
    const uint8_t value = out_volume->at(j);
    if (value == kEmpty) {
      out_volume->set(kUnknown, j);
    }
  }

  VoxFill(out_volume, kRayWindowSize);

  for (int j = 0; j < out_volume->size(); ++j) {
    const uint8_t value = out_volume->at(j);
    if (value == kVisible) {
      out_volume->set(kEmpty, j);
    }
  }

  for (int j = 0; j < out_volume->size(); ++j) {
    const uint8_t value = out_volume->at(j);
    if (value == kUnknown) {
      out_volume->set(kFilled, j);
    }
  }

}

// Visible pixels are empty.
void SweepXYZAndMarkVisible(VoxelGrid *voxels) {
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

void VoxelGrid::save(const std::string &filename) {
  const std::vector<int> shape{this->nx(), this->ny(), this->nz()};
  FileIO::SerializeTensor<uint8_t>(filename, this->data(), shape);
}

// `voxels` must be initialized with kUnknown values.
int VoxFill(VoxelGrid *voxels, int window_size) {
  SweepXYZAndMarkVisible(voxels);
  const int nx = voxels->nx();
  const int ny = voxels->ny();
  const int nz = voxels->nz();

  VoxelGrid directional_output(nx, ny, nz);

  const int start = -window_size / 2;
  const int end = window_size / 2;

  int count = 0;
  for (int vx = start; vx < end; vx += 1) {
    for (int vy = start; vy < end; vy += 1) {
      for (int vz = start; vz < end; vz += 1) {
        if (vx == start or vy == start or vz == start or vx == end - 1 or
            vy == end - 1 or vz == end - 1) {
          array<float, 3> direction{
              static_cast<float>(vx),
              static_cast<float>(vy),
              static_cast<float>(vz),
          };

          // memoized.
          for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
              for (int z = 0; z < nz; z++) {
                if (voxels->at(x, y, z) == kUnknown && directional_output.at(x, y, z) == kEmpty) {
                  RayBackTracing({x, y, z}, direction, *voxels, &directional_output);
                }
              }
            }
          }

          Ensures(directional_output.size() == voxels->size());
          for (int i = 0; i < directional_output.size(); i++) {
            if (directional_output.at(i) == kVisible) {
              voxels->set(kVisible, i);
            }
          }

          directional_output.reset();
          count++;
        }
      }
    }
  }

  return count;
}

// http://www.cse.chalmers.se/edu/year/2011/course/TDA361_Computer_Graphics/grid.pdf
bool RayBackTracing(const array<int, 3> &starting_pos, const array<float, 3> &direction,
                    const VoxelGrid &voxels, VoxelGrid *directional_output) {
  const int nx = voxels.nx();
  const int ny = voxels.ny();
  const int nz = voxels.nz();

  const int stepX = (direction[0] >= 0) ? 1 : -1;
  const int stepY = (direction[1] >= 0) ? 1 : -1;
  const int stepZ = (direction[2] >= 0) ? 1 : -1;

  const float tDeltaX = 1.0f / std::abs(direction[0]);
  const float tDeltaY = 1.0f / std::abs(direction[1]);
  const float tDeltaZ = 1.0f / std::abs(direction[2]);

  int x = starting_pos[0];
  int y = starting_pos[1];
  int z = starting_pos[2];

  float tMaxX = tDeltaX * 0.5f;
  float tMaxY = tDeltaY * 0.5f;
  float tMaxZ = tDeltaZ * 0.5f;

  int count = 0;

  bool is_visible = true;
  if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
    while (true) {
      if (directional_output->at(x, y, z) == kVisible) {
        break;
      } else if (directional_output->at(x, y, z) == kFilled or voxels.at(x, y, z) == kFilled) {
        is_visible = false;
        break;
      }

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          x += stepX;
          tMaxX += tDeltaX;
          if (x < 0 || x >= nx) {
            break;
          }
        } else {
          z += stepZ;
          tMaxZ += tDeltaZ;
          if (z < 0 || z >= nz) {
            break;
          }
        }
      } else {
        if (tMaxY < tMaxZ) {
          y += stepY;
          tMaxY += tDeltaY;
          if (y < 0 || y >= ny) {
            break;
          }
        } else {
          z += stepZ;
          tMaxZ += tDeltaZ;
          if (z < 0 || z >= nz) {
            break;
          }
        }
      }

      count++;
    }
  }

  x = starting_pos[0];
  y = starting_pos[1];
  z = starting_pos[2];

  tMaxX = tDeltaX * 0.5f;
  tMaxY = tDeltaY * 0.5f;
  tMaxZ = tDeltaZ * 0.5f;

  if (is_visible) {
    if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
      for (int i = 0; i <= count; i++) {
        directional_output->set(kVisible, x, y, z);

        if (tMaxX < tMaxY) {
          if (tMaxX < tMaxZ) {
            x += stepX;
            tMaxX += tDeltaX;
          } else {
            z += stepZ;
            tMaxZ += tDeltaZ;
          }
        } else {
          if (tMaxY < tMaxZ) {
            y += stepY;
            tMaxY += tDeltaY;
          } else {
            z += stepZ;
            tMaxZ += tDeltaZ;
          }
        }
      }
    }
  } else {
    for (int i = 0; i <= count; i++) {
      directional_output->set(kFilled, x, y, z);

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          x += stepX;
          tMaxX += tDeltaX;
        } else {
          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {
          y += stepY;
          tMaxY += tDeltaY;
        } else {
          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      }
    }
  }

  return is_visible;
}

}
}
