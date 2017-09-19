//
// Created by daeyun on 9/10/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <string>
#include <vector>

#include "cpp/lib/single_batch_loader.h"

// Interface for Python ctypes.
extern "C" {
uint32_t ReadSingleBatch_float32(const char **filenames,
                                 int32_t filenames_size,
                                 const int32_t *shape,
                                 int32_t shape_size,
                                 float *data);

uint32_t ReadSingleBatch_uint8(const char **filenames,
                               int32_t filenames_size,
                               const int32_t *shape,
                               int32_t shape_size,
                               uint8_t *data);

uint32_t ReadSingleBatch_int32(const char **filenames,
                               int32_t filenames_size,
                               const int32_t *shape,
                               int32_t shape_size,
                               int32_t *data);
}

template<typename T>
uint32_t ReadSingleBatch(const char **filenames,
                         const int32_t filenames_size,
                         const int32_t *shape,
                         const int32_t shape_size,
                         T *data) {
  std::vector<std::string> filenames_vector;
  filenames_vector.reserve((size_t) filenames_size);
  for (int i = 0; i < filenames_size; ++i) {
    // Maximum path size sanity check.
    filenames_vector.emplace_back(filenames[i]);
  }

  std::vector<int> shape_vector;
  shape_vector.reserve((size_t) shape_size);
  for (int i = 0; i < shape_size; ++i) {
    shape_vector.emplace_back(shape[i]);
  }

  return mvshape::ReadSingleBatch<T>(filenames_vector, shape_vector, data);
}

uint32_t ReadSingleBatch_float32(const char **filenames,
                                 const int32_t filenames_size,
                                 const int32_t *shape,
                                 const int32_t shape_size,
                                 float *data) {
  return ReadSingleBatch<float>(filenames, filenames_size, shape, shape_size, data);
}

uint32_t ReadSingleBatch_uint8(const char **filenames,
                               const int32_t filenames_size,
                               const int32_t *shape,
                               const int32_t shape_size,
                               uint8_t *data) {
  return ReadSingleBatch<uint8_t>(filenames, filenames_size, shape, shape_size, data);
}

uint32_t ReadSingleBatch_int32(const char **filenames,
                               const int32_t filenames_size,
                               const int32_t *shape,
                               const int32_t shape_size,
                               int32_t *data) {
  return ReadSingleBatch<int32_t>(filenames, filenames_size, shape, shape_size, data);
}
