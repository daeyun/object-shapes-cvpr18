#include "single_batch_loader.h"

#include "cpp/lib/file_io.h"
#include "benchmark.h"

namespace mvshape {

// `data` should be allocated by the caller.
// Returns microseconds.
template<typename T>
uint32_t ReadSingleBatch(const vector<string> &filenames, const vector<int> &shape, T *data) {
  if (shape.empty()) {
    throw std::runtime_error("Empty shape.");
  }

  const auto batch_size = static_cast<const int>(filenames.size());
  if (batch_size <= 0) {
    throw std::runtime_error("Batch size should be at least 1.");
  }

  size_t num_elements = 1;
  for (int i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      throw std::runtime_error("Dimension should be at least 1.");
    }
    if (i > 0) {
      num_elements *= shape[i];
    }
  }

  if (not FileIO::Exists(filenames[0])) {
    throw std::runtime_error("File not found: " + filenames[0]);
  }

  auto start = MicroSecondsSinceEpoch();
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < batch_size; ++i) {
    const string compressed = FileIO::ReadBytes(filenames[i]);
    string serialized;
    FileIO::DecompressBytes(compressed.data(), &serialized);

    const auto *header = reinterpret_cast<const int32_t *>(serialized.data());

    const int32_t dims = *header;
    size_t size = serialized.size() - sizeof(int32_t) * (dims + 1);
    if (GSL_UNLIKELY(num_elements * sizeof(T) != size)) {
      std::ostringstream stream;
      stream << "Tensor size mismatch: " << (num_elements * sizeof(T)) << ", " << size;
      throw std::runtime_error(stream.str());
    }

    const auto *data_i_start = reinterpret_cast<const T *>(header + dims + 1);
    std::memcpy(data + i * num_elements, data_i_start, num_elements * sizeof(T));
  }

  auto elapsed = MicroSecondsSinceEpoch() - start;

  return static_cast<uint32_t>(elapsed);
};

template uint32_t ReadSingleBatch<float>(const vector<string> &filenames, const vector<int> &shape, float *data);
template uint32_t ReadSingleBatch<int32_t>(const vector<string> &filenames, const vector<int> &shape, int32_t *data);
template uint32_t ReadSingleBatch<uint8_t>(const vector<string> &filenames, const vector<int> &shape, uint8_t *data);

}
