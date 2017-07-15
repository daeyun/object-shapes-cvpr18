//
// Created by daeyun on 6/14/17.
//
#include "file_io.h"

#include <iostream>
#include <string>

#include <fstream>
#include <chrono>
#include <type_traits>
#include <sys/stat.h>

#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <blosc.h>
#include <boost/filesystem.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "benchmark.h"
#include "flags.h"

namespace mvshape {
namespace FileIO {
namespace fs = boost::filesystem;

// Dumps raw bytes to a stream.
void WriteBytes(const void *src, size_t size_bytes, std::ostream *stream) {
  stream->write(reinterpret_cast<const char *>(src), size_bytes);
}

void WriteBytes(const std::string &src, std::ostream *stream) {
  WriteBytes(src.data(), src.size(), stream);
}

template<typename T>
void WriteBytes(const vector<T> &src, std::ostream *stream) {
  WriteBytes(src.data(), sizeof(T) * src.size(), stream);
}

template<typename T>
void WriteBytes(const T &src, std::ostream *stream) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
  WriteBytes(&src, sizeof(src), stream);
}

template<>
void WriteBytes(const vector<cv::Mat> &src, std::ostream *stream) {
  for (const auto &item : src) {
    Expects(item.isContinuous());
    auto size = item.total() * item.elemSize();
    WriteBytes(item.data, size, stream);
  }
}

template<typename T>
std::string ToBytes(const vector<T> &values) {
  std::ostringstream stream;
  WriteBytes(values.data(), sizeof(T) * values.size(), &stream);
  return stream.str();
}

template std::string ToBytes<float>(const vector<float> &values);

std::string ReadBytes(const std::string &path) {
  auto canonical_path = boost::filesystem::canonical(path).string();
  std::ifstream stream(canonical_path);
  std::string content((std::istreambuf_iterator<char>(stream)),
                      (std::istreambuf_iterator<char>()));
  return content;
}

void CompressBytes(const void *src, int size_bytes, const char *compressor, int level, size_t typesize,
                   std::string *out) {
  out->resize(std::max(static_cast<size_t>(size_bytes * 4 + BLOSC_MAX_OVERHEAD), static_cast<size_t>(512)));
  int compressed_size = blosc_compress_ctx(level, true, typesize, static_cast<size_t>(size_bytes),
                                           src, &(*out)[0], out->size(), compressor, 0, 1);
  out->resize(static_cast<size_t>(compressed_size));
  if (compressed_size <= 0) {
    throw std::runtime_error("Compression failed.");
  }
}

void DecompressBytes(const void *src, std::string *out) {
  size_t nbytes, cbytes, blocksize;
  blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
  out->resize(nbytes);
  int decompressed_size = blosc_decompress_ctx(src, &(*out)[0], out->size(), 1);
  if (decompressed_size <= 0) {
    throw std::runtime_error("Deompression failed.");
  }
}

template<typename T>
void SerializeTensor(const std::string &filename, const void *data, const std::vector<int> &shape) {
  const int num_bytes = sizeof(T) * std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());

  Expects(num_bytes > sizeof(T));

  std::ostringstream stream;
  WriteBytes<int32_t>(shape.size(), &stream);
  WriteBytes<int32_t>(shape, &stream);
  WriteBytes(data, num_bytes, &stream);

  string encoded = stream.str();

  std::string compressed;
  CompressBytes(encoded.data(), static_cast<int>(encoded.size()), "lz4hc", 6, sizeof(float), &compressed);
  void *out_ptr = &compressed[0];
  auto out_size = compressed.size();

  std::ofstream file;
  auto absolute_path = boost::filesystem::absolute(filename).string();
  PrepareDir(fs::path(absolute_path).parent_path().string());

  LOG(INFO) << "Saving " << absolute_path;
  file.open(absolute_path, std::ios_base::out | std::ios_base::binary);

  WriteBytes(out_ptr, out_size, &file);
  Ensures(Exists(absolute_path));
}

string JoinPath(const string &a, const string &b) {
  return (fs::path(a) / fs::path(b)).string();
}

void RemoveDirIfExists(const string &path) {
  if (fs::is_directory(path)) {
    DLOG(INFO) << "rm -rf " << fs::absolute(path);
    fs::remove_all(path);
  }
}

vector<string> RegularFilesInDirectory(const string &dir) {
  fs::path path(dir);
  vector<string> paths;
  if (fs::exists(path) && fs::is_directory(path)) {
    fs::directory_iterator end_iter;
    for (fs::directory_iterator dir_iter(path); dir_iter != end_iter; ++dir_iter) {
      if (fs::is_regular_file(dir_iter->status())) {
        paths.push_back(dir_iter->path().string());
      }
    }
  }
  std::sort(std::begin(paths), std::end(paths));

  return paths;
}

string WriteIndexFile(const string &dirname, int index) {
  // TODO
  std::ofstream file;
  auto absolute_dir_path = boost::filesystem::absolute(dirname).string();
  PrepareDir(absolute_dir_path);
  auto filename = (fs::path(absolute_dir_path) / ".index").string();
  file.open(filename, std::ios_base::out);
  file << index;
  file.close();
  return filename;
}

template void SerializeTensor<float>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<double>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<char>(const std::string &filename, const void *data, const std::vector<int> &shape);
template void SerializeTensor<int>(const std::string &filename, const void *data, const std::vector<int> &shape);

void SerializeImages(const std::string &filename, const vector<cv::Mat> &images, bool compress, bool append) {
  for (const auto &image : images) {
    Expects(image.size == images[0].size);
    Expects(image.elemSize() == images[0].elemSize());
    Expects(image.type() == images[0].type());
  }
  PrepareDir(fs::path(filename).parent_path().string());

  int image_type;
  switch (images[0].channels()) {
    case 1:
      image_type = CV_32FC1;
      break;
    case 2:
      image_type = CV_32FC2;
      break;
    case 3:
      image_type = CV_32FC3;
      break;
    case 4:
      image_type = CV_32FC4;
      break;
    default:
      throw std::runtime_error("Unexpected number of channels: " + std::to_string(images[0].channels()));
  }

  vector<cv::Mat> images_float;
  for (int j = 0; j < images.size(); ++j) {
    images_float.emplace_back();
    images[j].convertTo(images_float[j], image_type);
    Expects(images_float[j].isContinuous());
  }

  std::ofstream file;
  auto absolute_path = boost::filesystem::absolute(filename).string();

#if (!NDEBUG)
  DLOG(INFO) << "Saving " << images.size() << " float arrays of shape: (" << images[0].size[0] << ", "
             << images[0].size[1] << ", " << images[0].channels() << ") " << "Path: " << absolute_path;
  auto start_time = MicroSecondsSinceEpoch();
#endif

  std::ostringstream stream;

  vector<int32_t> shape;

  shape.push_back(images_float.size());
  for (int i = 0; i < images_float[0].dims; ++i) {
    shape.push_back(images_float[0].size[i]);
  }
  shape.push_back(static_cast<int32_t>(images_float[0].channels()));

  WriteBytes<int32_t>(shape.size(), &stream);
  WriteBytes<int32_t>(shape, &stream);
  WriteBytes<cv::Mat>(images, &stream);

  string encoded = stream.str();

  char *out_ptr;
  size_t out_size;
  std::string compressed;
  if (compress) {
    CompressBytes(encoded.data(), static_cast<int>(encoded.size()), "lz4hc", 6, sizeof(float), &compressed);
    out_ptr = &compressed[0];
    out_size = compressed.size();
  } else {
    out_ptr = &encoded[0];
    out_size = encoded.size();
  }

#if (!NDEBUG)
  auto write_start_time = MicroSecondsSinceEpoch();
#endif

  file.open(absolute_path, (append ? std::ios_base::app : std::ios_base::out) | std::ios_base::binary);

  WriteBytes(out_ptr, out_size, &file);

#if (!NDEBUG)
  auto now = MicroSecondsSinceEpoch();
  DLOG(INFO) << "Time elapsed - IO: " << (now - write_start_time) / 1000.0 << " ms"
             << ", Compression: " << (write_start_time - start_time) / 1000.0 << " ms";
  auto size_on_disk = boost::filesystem::file_size(absolute_path) / 1024.0;
  DLOG(INFO) << "Size on disk: " << size_on_disk << " KB. Compression ratio: "
             << size_on_disk / (encoded.size() / 1000.0);
#endif
}

bool ReadTriangles(const std::string &filename,
                   const std::function<void(const std::array<std::array<float, 3>, 3> &)> &triangle_handler) {
  LOG(INFO) << "Importing " << filename;
  Expects(boost::filesystem::exists(filename));

  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate);

  if (!scene) {
    LOG(ERROR) << importer.GetErrorString();
    return false;
  }

  int triangle_count = 0;
  for (int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    for (int j = 0; j < mesh->mNumFaces; ++j) {
      auto face = mesh->mFaces[j];
      Expects(face.mNumIndices == 3);

      for (int k = 0; k < 3; ++k) {
        if (face.mIndices[k] >= mesh->mNumVertices) {
          LOG(WARNING) << "Invalid vertex index found. Skipping.";
          continue;
        }
      }
      auto a = mesh->mVertices[face.mIndices[0]];
      auto b = mesh->mVertices[face.mIndices[1]];
      auto c = mesh->mVertices[face.mIndices[2]];
      triangle_handler({std::array<float, 3>{a.x, a.y, a.z},
                        std::array<float, 3>{b.x, b.y, b.z},
                        std::array<float, 3>{c.x, c.y, c.z}});
      ++triangle_count;
    }
  }

  if (triangle_count <= 0) {
    LOG(WARNING) << "No triangles found in mesh file.";
  }

  return true;
}

vector<array<array<float, 3>, 3>> ReadTriangles(const std::string &filename) {
  vector<array<array<float, 3>, 3>> triangles;
  ReadTriangles(filename,
                [&](const array<array<float, 3>, 3> triangle) {
                  triangles.push_back(triangle);
                });
  return triangles;
}

bool Exists(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

string SystemTempDir() {
  return fs::temp_directory_path().string();
}

bool PrepareDir(const string &filename) {
  auto path = fs::absolute(filename);
  if (!fs::is_directory(path)) {
    Expects(!fs::is_regular_file(path));
    fs::create_directories(path);
    DLOG(INFO) << "mkdir -p " << path.string();
    return true;
  }
  return false;
}

string NamedEmptyTempDir(const string &name) {
  string path = (fs::path(SystemTempDir()) / name).string();

  Expects(!fs::is_regular_file(path));

  RemoveDirIfExists(path);
  fs::create_directories(path);

  DLOG(INFO) << "Using temp directory " << path;
  return path;
}

string FullDataPath(const fs::path &path) {
  auto path_string = path.string();
  auto data_dir = fs::absolute(FLAGS_data_dir);
  if (path_string.size() == 0 || path_string == "/") {
    return data_dir.string();
  }
  string p = path_string;
  if (p[0] == '/') {
    p.erase(0, 1);
  }
  return (data_dir / fs::path(p)).string();
}

string FullOutPath(const fs::path &path) {
  auto path_string = path.string();
  auto data_dir = fs::absolute(FLAGS_out_dir);
  if (path_string.size() == 0 || path_string == "/") {
    return data_dir.string();
  }
  string p = path_string;
  if (p[0] == '/') {
    p.erase(0, 1);
  }
  return (data_dir / fs::path(p)).string();
}

}  // namespace FileIO
}  // namespace mvshape
