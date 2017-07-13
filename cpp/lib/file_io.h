//
// Created by daeyun on 6/7/17.
//

#pragma once

#include <string>
#include <functional>

#include <opencv2/opencv.hpp>

#include "common.h"

namespace mvshape {
namespace FileIO {

/**
 * Dumps raw bytes to a file.
 * @param src
 * @param size_bytes
 * @param file
 */

void WriteBytes(const std::string &src, std::ostream *stream);

void WriteBytes(const std::string &src, std::ostream *stream);

template<typename T>
void WriteBytes(const vector<T> &src, std::ostream *stream);

// T must be trivially copyable.
template<typename T>
void WriteBytes(const T &src, std::ostream *stream);

template<typename T>
std::string ToBytes(const vector<T> &values);

std::string ReadBytes(const std::string &path);

/**
 * Compresses a binary string.
 * @param[in] src
 * @param[in] size_bytes The number of bytes in `src` buffer.
 * @param[in] compressor One of "lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd".
 * @param[in] level A number between 0 (no compression) and 9 (maximum compression).
 * @param[in] typesize The number of bytes for the atomic type in binary `src` buffer. e.g. sizeof(float)
 * @param[out] out
 * @throw std::runtime_error
 */
void CompressBytes(const void *src, int size_bytes, const char *compressor, int level, size_t typesize,
                   std::string *out);

/**
 * The inverse of `CompressBytes`.
 * @param[in] src
 * @param[out] out
 * @throw std::runtime_error
 */
void DecompressBytes(const void *src, std::string *out);

/**
 * Saves an OpenCV array to a binary file.
 * Format: int32 array shape values [n, dim_1, dim_2, ..., dim_n]
 *         followed by [data]
 * @param filename
 * @param images float image arrays of the same size and type.
 * @throw std::runtime_error
 */
void SerializeImages(const std::string &filename,
                     const vector<cv::Mat> &images,
                     bool compress = true,
                     bool append = false);

template<typename T>
void SerializeTensor(const std::string &filename, const void *data, const std::vector<int> &shape);

bool ReadTriangles(const std::string &filename,
                   const std::function<void(const std::array<std::array<float, 3>, 3> &)> &triangle_handler);

vector<array<array<float, 3>, 3>> ReadTriangles(const std::string &filename);

string WriteIndexFile(const string &dirname, int index);

bool Exists(const std::string &filename);

string SystemTempDir();

string NewTempDir(const string &name);

string FullDataPath(const string &path);

string FullOutPath(const string &path);

bool PrepareDir(const string &filename);

vector<string> RegularFilesInDirectory(const string &dir);

}  // namespace FileIO
}  // namespace mvshape
