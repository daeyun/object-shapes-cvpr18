#include "single_batch_loader.h"

#include <sys/stat.h>
#include <chrono>
#include <fstream>

#include <png.h>
#include <blosc.h>

namespace mvshape {

// from file_io.cc
namespace mvshape_lite {
namespace FileIO {
bool Exists(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}
std::string ReadBytes(const std::string &path) {
  std::ifstream stream(path);
  std::string content((std::istreambuf_iterator<char>(stream)),
                      (std::istreambuf_iterator<char>()));
  return content;
}
void DecompressBytes(const void *src, std::string *out) {
  size_t nbytes, cbytes, blocksize;
  blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
  out->resize(nbytes);
  int decompressed_size = blosc_decompress_ctx(src, &(*out)[0], out->size(), 1);
  if (decompressed_size <= 0) {
    std::cerr << "ERROR: Decompression failed." << std::endl;
    throw std::runtime_error("Decompression failed.");
  }
}
}
// from benchmark.cc
long MicroSecondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::microseconds>
      (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
}

// beginning of code unique to this file.

void ReadPNG(const std::string &filename, std::string *png_bytes) {
  // https://gist.github.com/niw/5963798
  FILE *fp = fopen(filename.data(), "rb");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    throw std::runtime_error("Error in png_create_read_struct");
  }

  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, &info, nullptr);
    throw std::runtime_error("Error in png_create_info_struct");
  }

  if (setjmp(png_jmpbuf(png))) throw std::runtime_error("Error in setjmp(png_jmpbuf(png))");

  png_init_io(png, fp);
  png_read_info(png, info);

  auto width = static_cast<size_t>(png_get_image_width(png, info));
  auto height = static_cast<size_t>(png_get_image_height(png, info));
  uint8_t color_type = png_get_color_type(png, info);
  uint8_t bit_depth = png_get_bit_depth(png, info);

  if (bit_depth != 8) {
    throw std::runtime_error("Only 8 bit PNG images are supported for now.");
  }

  if (color_type != PNG_COLOR_TYPE_RGB) {
    // look at the gist code for RGBA.
    throw std::runtime_error("Only RGB PNG images are supported for now.");
  }

  png_read_update_info(png, info);

  auto *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);

  // `num_row_bytes` should be 3*width
  auto num_row_bytes = static_cast<uint32_t>(png_get_rowbytes(png, info));
  auto buffer_size = height * num_row_bytes;
  auto *image_buffer = (png_byte *) malloc(buffer_size);

  for (int y = 0; y < height; ++y) {
    row_pointers[y] = image_buffer + y * num_row_bytes;
  }

  png_read_image(png, row_pointers);

  fclose(fp);
  free(row_pointers);
  png_destroy_read_struct(&png, &info, nullptr);
  png = nullptr;
  info = nullptr;

  // `image_buffer` contains h*w*3 bytes.
  // transpose to 3*h*w and write `png_bytes`.

  png_bytes->resize(buffer_size);

  const auto wh = width * height;
  for (size_t j = 0; j < wh; ++j) {
    const auto offset = j * 3;
    png_bytes->at(j) = image_buffer[offset];
    png_bytes->at(j + wh) = image_buffer[offset + 1];
    png_bytes->at(j + 2 * wh) = image_buffer[offset + 2];
  }
  free(image_buffer);
}

bool EndsWith(std::string const &fullString, std::string const &ending) {
  // https://stackoverflow.com/a/874160
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

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

  if (not mvshape_lite::FileIO::Exists(filenames[0])) {
    throw std::runtime_error("File not found: " + filenames[0]);
  }

  if (EndsWith(filenames[0], ".png")) {
    if (!std::is_same<uint8_t, T>::value) {
      throw std::runtime_error("Tensor data type must be uint8 for png files.");
    }
    if (shape[shape.size() - 3] != 3) {
      throw std::runtime_error("Tensor shape must be (..., 3, H, W) for png files. Only RGB is supported for now.");
    }
  }

  auto start = mvshape_lite::MicroSecondsSinceEpoch();
#pragma omp parallel for schedule(dynamic) num_threads(6)
  for (int i = 0; i < batch_size; ++i) {
    const auto &filename = filenames[i];

    std::string buffer;
    const T *data_i_start;  // points to somewhere in `buffer`
    size_t size;

    // TODO: make sure all filenames have the same extension.
    if (EndsWith(filename, ".png")) {
      ReadPNG(filename, &buffer);
      size = buffer.size();
      data_i_start = reinterpret_cast<const T *>(buffer.data());
    } else {
      const string compressed = mvshape_lite::FileIO::ReadBytes(filename);
      mvshape_lite::FileIO::DecompressBytes(compressed.data(), &buffer);
      const auto *header = reinterpret_cast<const int32_t *>(buffer.data());
      const int32_t dims = *header;
      size = buffer.size() - sizeof(int32_t) * (dims + 1);
      data_i_start = reinterpret_cast<const T *>(header + dims + 1);
    }

    if (GSL_UNLIKELY(num_elements * sizeof(T) != size)) {
      std::ostringstream stream;
      stream << "Tensor size mismatch: " << (num_elements * sizeof(T)) << ", " << size;
      throw std::runtime_error(stream.str());
    }

//    std::memcpy(data + i * num_elements, data_i_start, num_elements * sizeof(T));
    std::copy(data_i_start, data_i_start + num_elements, data + i * num_elements);
  }

  auto elapsed = mvshape_lite::MicroSecondsSinceEpoch() - start;

  return static_cast<uint32_t>(elapsed);
}

template uint32_t ReadSingleBatch<float>(const vector<string> &filenames, const vector<int> &shape, float *data);
template uint32_t ReadSingleBatch<int32_t>(const vector<string> &filenames, const vector<int> &shape, int32_t *data);
template uint32_t ReadSingleBatch<uint8_t>(const vector<string> &filenames, const vector<int> &shape, uint8_t *data);

}
