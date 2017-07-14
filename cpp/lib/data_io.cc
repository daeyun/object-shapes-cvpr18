//
// Created by daeyun on 6/29/17.
//

#include "data_io.h"

#include <sys/wait.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>
#include <cpp_lru_cache/lrucache.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <fstream>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <iomanip>

#include "common.h"
#include "database.h"
#include "file_io.h"
#include "transforms.h"
#include "egl_rendering.h"
#include "benchmark.h"
#include "multiprocessing.h"
#include "random_utils.h"

DEFINE_int32(num_processes, 1, "Number of processes.");

namespace mvshape {
namespace Data {

namespace mv = mvshape_dataset;
namespace fs = boost::filesystem;

constexpr float kNear = 0.1;
constexpr float kFar = 40;
constexpr int kImageNormalizationUpScale = 4;
constexpr int kNormalizationPadding = 1;

constexpr int kNumReaderThreads = 4;
constexpr int kQueueSizeMultiplier = 3;
constexpr int kSlowIOWarningMicroSec = 10000;
constexpr int kThreadJoinWaitSeconds = 5;

void RenderImages(const vector<const mv::Rendering *> &renderables) {
  auto sorted_renderables = vector<const mv::Rendering *>(renderables.begin(), renderables.end());

  // Increasing order.
  std::sort(sorted_renderables.begin(),
            sorted_renderables.end(),
            [](const mv::Rendering *a, const mv::Rendering *b) -> bool {
              auto ar = a->resolution();
              auto br = b->resolution();
              if (a->is_normalized()) {
                ar *= kImageNormalizationUpScale;
              }
              if (b->is_normalized()) {
                br *= kImageNormalizationUpScale;
              }
              return ar < br && a->mesh_filename() < b->mesh_filename();
            });

  int prev_resolution = -1;
  Rendering::Renderer *renderer = nullptr;

  auto cached_renderer = [&](int resolution) -> Rendering::Renderer * {
    Expects(resolution > 1);
    if (resolution != prev_resolution) {
      Rendering::RendererConfig render_config{
          .width = resolution,
          .height = resolution,
      };

      if (renderer != nullptr) {
        renderer->Cleanup();
        delete renderer;
      }

      // TODO: avoid using new
      renderer = new Rendering::ShapeRenderer(render_config);
      prev_resolution = resolution;
    }
    return renderer;
  };
  cache::lru_cache<string, shared_ptr<vector<array<array<float, 3>, 3>>>> triangles_cache(500);
  auto read_triangles = [&](const string &filename) {
    if (!triangles_cache.exists(filename)) {
      auto triangles = make_shared<vector<array<array<float, 3>, 3>>>(FileIO::ReadTriangles(filename));
      triangles_cache.put(filename, triangles);
    }
    return triangles_cache.get(filename);
  };
  string prev_mesh_filename;

  Timer timer("rendering");

  pair<int, int> start_end;
  MP::MultiFork(sorted_renderables.size(), FLAGS_num_processes, &start_end);

  for (int i = start_end.first; i < start_end.second; ++i) {
    const mv::Rendering *renderable = sorted_renderables[i];

    if (renderable->rendering_type() == mv::Rendering_Type_RGB
        or renderable->rendering_type() == mv::Rendering_Type_NORMAL) {
      // TODO: RGB not implemented.
      // TODO: Normals are ignored for now.
      continue;
    }

    string out_filename = FileIO::FullOutPath(renderable->filename());
    if (FileIO::Exists(out_filename)) {
      LOG_EVERY_N(INFO, 100) << "Already exists: " << out_filename;
      continue;
    }

    LOG_EVERY_N(INFO, 100) << Data::ToString(*renderable);
    timer.Toc();

    string mesh_filename = FileIO::FullDataPath(renderable->mesh_filename());

    int resolution = renderable->resolution();
    if (renderable->is_normalized()) {
      resolution *= kImageNormalizationUpScale;
    }
    renderer = cached_renderer(resolution);

    if (mesh_filename != prev_mesh_filename or !renderer->has_triangles()) {
      const auto triangles = read_triangles(mesh_filename);
      renderer->SetTriangleVertices(*triangles);
      prev_mesh_filename = mesh_filename;
    }

    float scale = 1;
    if (renderable->scale() != 0) {
      scale = renderable->scale();
    }
    unique_ptr<Camera> camera;

    Expects(renderable->eye_size() == 3);
    Expects(renderable->up_size() == 3);
    Expects(renderable->lookat_size() == 3);
    Vec3 eye{renderable->eye(0), renderable->eye(1), renderable->eye(2)};
    Vec3 up{renderable->up(0), renderable->up(1), renderable->up(2)};
    Vec3 lookat{renderable->lookat(0), renderable->lookat(1), renderable->lookat(2)};

    if (renderable->fov() < 1) {
      FrustumParams frustum;
      frustum.near = kNear;
      frustum.far = kFar;
      camera = make_unique<OrthographicCamera>(eye, lookat, up, frustum);
    } else {
      FrustumParams frustum = mvshape::FrustumParams::MakePerspective(renderable->fov(), 1, kNear, kFar);
      camera = make_unique<PerspectiveCamera>(eye, lookat, up, frustum);
    }

    // TODO
    // Specific to shrec12
    Expects(std::abs(15 - (camera->position() - camera->lookat_position()).norm()) < 1e-2);

    vector<unique_ptr<Camera>> cams;

    // Populate output cameras.
    if (renderable->set_size() == 1) {
      // If a single image is requested, use the camera as-is.
      cams.push_back(std::move(camera));
    } else if (renderable->set_size() == 6) {
      // If multiple images is requested, derive the cameras from the reference camera.
      // The reference camera does not have to be part of the 6.
      SixViews(*camera, &cams);
    } else {
      throw std::runtime_error("Not implemented");
    }

    vector<mvshape::Rendering::FrameConfig> configs;
    for (int j = 0; j < cams.size(); ++j) {
      configs.push_back(
          mvshape::Rendering::FrameConfig {
              .camera = cams[j].get(),
              .scale = scale,
          }
      );
    }

    vector<vector<unique_ptr<cv::Mat>>> out_frames;
    renderer->Render(configs, &out_frames);

    int out_channel = 0;
    vector<cv::Mat> out_images;
    if (renderable->rendering_type() == mv::Rendering_Type_DEPTH) {
      out_channel = 0;
    } else if (renderable->rendering_type() == mv::Rendering_Type_NORMAL) {
      out_channel = 1;
    } else {
      throw std::runtime_error("Not implemented");
    }

    for (int j = 0; j < out_frames.size(); ++j) {
      out_images.push_back(*out_frames[j][out_channel]);
    }

    if (renderable->is_normalized()) {
      for (int k = 0; k < out_images.size(); ++k) {
        mvshape::RescaleAndRecenter(out_images[k],
                                    tuple<int, int>{renderable->resolution(), renderable->resolution()},
                                    kNormalizationPadding, &out_images[k]);
      }
    } else {
      for (int k = 0; k < out_images.size(); ++k) {
        mvshape::ReplaceZerosWithBackgroundValue(&out_images[k]);
      }
    }

    FileIO::SerializeImages(out_filename, out_images,
                            true,  // Compress
                            false);

  }

  if (renderer != nullptr) {
    renderer->Cleanup();
    delete renderer;
    renderer = nullptr;
  }

  MP::Join();

  LOG(INFO) << "Rendering done";
}

void Voxelize(const vector<const mv::Rendering *> &renderable) {

}

void GenerateDataset() {
  vector<mv::Rendering> renderables;
  int count = Data::ReadRenderables("database/shrec12.sqlite", "shrec12_renderings", &renderables);
  Expects(0 < count);

  vector<const mv::Rendering *> renderables_2d;
  vector<const mv::Rendering *> renderables_3d;
  for (const auto &renderable : renderables) {
    if (renderable.rendering_type() == mv::Rendering_Type_VOXELS) {
      renderables_3d.push_back(&renderable);
    } else {
      renderables_2d.push_back(&renderable);
    }
  }

  RenderImages(renderables_2d);
  Voxelize(renderables_3d);

}

void SixViews(const Camera &camera, vector<unique_ptr<Camera>> *out) {
  auto left = camera.up().cross(camera.viewing_direction());
  vector<Eigen::Transform<double, 3, Eigen::Affine>> transformations;

  // Front
  transformations.emplace_back(Eigen::Transform<double, 3, Eigen::Affine>::Identity());
  // Back
  transformations.emplace_back(Eigen::AngleAxis<double>(M_PI, camera.up()));
  // Left
  transformations.emplace_back(Eigen::AngleAxis<double>(-M_PI_2, camera.up()));
  // Right
  transformations.emplace_back(Eigen::AngleAxis<double>(M_PI_2, camera.up()));
  // Top
  transformations.emplace_back(Eigen::AngleAxis<double>(M_PI_2, left));
  // Bottom
  transformations.emplace_back(Eigen::AngleAxis<double>(M_PI, Eigen::AngleAxis<double>(M_PI_2, left) * camera.up())
                                   * Eigen::AngleAxis<double>(M_PI_2, left));

  for (const auto &transformation : transformations) {
    Vec3 pos = transformation * camera.position();
    Vec3 up = transformation * camera.up();
//    auto frustum = FrustumParams::MakePerspective(20, 1, 0.1, 40);
//    auto cam = make_unique<PerspectiveCamera>(pos, camera.lookat_position(), up, frustum);
    auto frustum = FrustumParams();
    frustum.near = kNear;
    frustum.far = kFar;
    auto cam = make_unique<OrthographicCamera>(pos, camera.lookat_position(), up, frustum);
    out->push_back(std::move(cam));
  }

}

int SaveExamples(const string &sqlite_file, const string &view_name, const string &out_dir) {
  std::map<mv::Split, mv::Examples> data;
  // FLAGS_data_dir is set to resources directory in test_main.cc.
  int count = mvshape::Data::ReadExamples(sqlite_file, view_name, &data);
  LOG(INFO) << count << " examples total";

  int save_count = 0;

  for (const auto &kv: data) {
    auto split = kv.first;
    auto examples = kv.second;
    string serialized = examples.SerializeAsString();

    string compressed;
    mvshape::FileIO::CompressBytes(serialized.data(), serialized.size(), "lz4hc", 7, sizeof(char), &compressed);
    string filename = boost::filesystem::absolute(fs::path(FileIO::FullOutPath(out_dir)) /
        boost::algorithm::to_lower_copy(mv::Split_Name(split))).string() + ".bin";

    FileIO::PrepareDir(fs::path(filename).parent_path().string());

    LOG(INFO) << "Saving " << examples.examples_size() << " in " << filename;
    save_count += examples.examples_size();

    std::ofstream file;
    file.open(filename, std::ios_base::out | std::ios_base::binary);
    mvshape::FileIO::WriteBytes(compressed, &file);
    file.close();
  }

  return save_count;
}

int LoadExamples(const string &filename, mv::Examples *examples) {
  string compressed = mvshape::FileIO::ReadBytes(filename);
  string serialized;
  mvshape::FileIO::DecompressBytes(compressed.data(), &serialized);
  mv::Examples parsed;
  parsed.ParseFromString(serialized);
  // Singular fields are overwritten. Repeated fields are appended.
  examples->MergeFrom(parsed);
  return parsed.examples_size();
}

int LoadExamples(const vector<string> &filenames, mv::Examples *examples) {
  int count = 0;
  for (int i = 0; i < filenames.size(); ++i) {
    const auto &filename = filenames[i];
    count += LoadExamples(filename, examples);
  }
  return count;
}

void LoadRenderingContent(const mv::Rendering &rendering, string *data) {
  const string path = FileIO::FullDataPath(rendering.filename());
  LoadRenderingContent(path, data);

  const int v = rendering.set_size();
  const int r = rendering.resolution();

  switch (rendering.rendering_type()) {
    case mv::Rendering_Type_DEPTH:
      Ensures(data->size() == 4 * r * r * v);
      break;
    case mv::Rendering_Type_NORMAL:
    case mv::Rendering_Type_RGB:
      Ensures(data->size() == 4 * r * r * v * 3);
      break;
    case mv::Rendering_Type_VOXELS:
      Ensures(data->size() == r * r * r * v);
      break;
    default:
      LOG(ERROR) << "Unknown rendering type: " << rendering.rendering_type();
      throw std::runtime_error("Unknown rendering type");
  }
}

void LoadRenderingContent(const string &full_filename, string *data) {
  const string compressed = FileIO::ReadBytes(full_filename);
  string serialized;
  FileIO::DecompressBytes(compressed.data(), &serialized);

  const int32_t *header = reinterpret_cast<const int32_t *>(serialized.data());

  const int32_t dims = *header;
  vector<int32_t> shape;
  for (int i = 1; i <= dims; ++i) {
    shape.push_back(*(header + i));
  }

  size_t size = serialized.size() - sizeof(int32_t) * (dims + 1);
  *data = string(reinterpret_cast<const char *>(header + dims + 1), size);
}

void RenderingReader::Read(int row_index,
                           const vector<int> &field_ids,
                           std::unordered_map<int, unique_ptr<string>> *field_data) {
  for (int i = 0; i < field_ids.size(); ++i) {
    const int field_id = field_ids[i];
    const auto &example = examples_->examples(row_index);
    auto *field_descriptor = example.descriptor()->FindFieldByNumber(field_id);
    auto *reflection = example.GetReflection();
    Expects(reflection->HasField(example, field_descriptor));
    auto rendering = dynamic_cast<const mv::Rendering &>(reflection->GetMessage(example, field_descriptor));
    (*field_data)[field_id] = make_unique<string>();
    LoadRenderingContent(rendering, (*field_data)[field_id].get());
  }
}

vector<int> RenderingReader::LoadBatch(const vector<int> &field_ids,
                                       int batch_size,
                                       std::unordered_map<int, vector<unique_ptr<string>>> *field_data) {
  vector<int> random_inds;
  RandomIndices(batch_size, &random_inds);
  for (const auto &ind : random_inds) {
    std::unordered_map<int, unique_ptr<string>> row_field_data;
    Read(ind, field_ids, &row_field_data);
    for (const auto &field_id : field_ids) {
      (*field_data)[field_id].emplace_back(std::move(row_field_data.at(field_id)));
    }
  }
  return random_inds;
}

void RenderingReader::PrepareIndices() {
  if (indices_.size() != examples_->examples_size()) {
    indices_.clear();
    indices_.reserve(examples_->examples_size());
    for (int i = 0; i < examples_->examples_size(); ++i) {
      indices_.push_back(i);
    }
  }
}

void RenderingReader::RandomIndices(int n, vector<int> *indices) {
  PrepareIndices();
  Random::ChooseN<int>(n, &indices_, indices);
}

BatchLoader::BatchLoader(const mv::Examples *examples,
                         const vector<int> &field_ids,
                         int batch_size,
                         bool is_seamless)
    : reader_({examples}),
      field_ids_(field_ids),
      batch_size_(batch_size),
      is_seamless_(is_seamless),
      queue_(nullptr),
      num_examples_dequeued_(0),
      num_examples_enqueued_(0) {
  indices_.resize(static_cast<size_t>(examples->examples_size()));
  std::iota(std::begin(indices_), std::end(indices_), 0); // Fill with 0, 1, ..., n.
  Random::Shuffle(indices_.begin(), indices_.end());
  Ensures(indices_.size() == examples->examples_size());

  StartWorkers();
}

void BatchLoader::DataReaderRoutine(int thread_id) {
  LOG(INFO) << "Starting DataReaderRoutine()";
  int local_count = 0;

  while (true) {
    int i;
    {
      std::lock_guard<std::mutex> lock(lock_);
      if (current_read_index_ >= indices_.size()) {
        if (is_seamless_) {
          Random::Shuffle(indices_.begin(), indices_.end());
        } else {
          break;
        }
        current_read_index_ = 0;
      }
      i = current_read_index_;
      current_read_index_++;
    }

    int row_index = indices_[i];
    std::unordered_map<int, unique_ptr<string>> field_data;
    reader_.Read(row_index, field_ids_, &field_data);

    BatchData single_example;
    for (int field_id : field_ids_) {
      single_example.file_fields[field_id] = *field_data[field_id];
    }
    single_example.size = 1;
    single_example.example_indices.emplace_back(row_index);
    if (queue_->Enqueue(single_example)) {
      local_count++;
      num_examples_enqueued_++;
    } else {
      break;
    }
  }

  if (num_examples_enqueued_.load() >= size()) {
    queue_->Close();
    if (!is_seamless_) {
      Ensures(num_examples_enqueued_.load() == size());
    }
  }

  LOG(INFO) << "End of DataReaderRoutine() after inserting " << local_count << " examples to the queue.";
}

void BatchLoader::BatchRoutine(int thread_id) {
  LOG(INFO) << "Starting BatchRoutine()";

  std::vector<BatchData> data(static_cast<size_t>(batch_size_));
  while (true) {
    size_t num_dequeued = queue_->Dequeue(std::begin(data), static_cast<size_t>(batch_size_));

    if (num_dequeued == 0) {
      // Queue was closed.
      break;
    }

    // Truncates if this is the last batch and there are fewer remaining items than the batch size.
    if (num_dequeued < batch_size_) {
      data.resize(num_dequeued);
    } else {
      Ensures(num_dequeued == batch_size_);
    }
    Ensures(num_dequeued == data.size());

    auto batch_data = std::make_unique<BatchData>();

    vector<size_t> field_sizes;
    for (int q = 0; q < field_ids_.size(); ++q) {
      int field_id = field_ids_[q];
      // Assume field sizes are the same in all examples.
      auto field_size = data[0].file_fields.at(field_id).size();
      batch_data->file_fields[field_id].reserve(field_size * num_dequeued);
      field_sizes.push_back(field_size);
    }

    for (int j = 0; j < data.size(); ++j) {
      const auto &item = data[j];
      batch_data->size += item.size;
      batch_data->example_indices.insert(
          batch_data->example_indices.end(),
          item.example_indices.begin(),
          item.example_indices.end());
      for (int q = 0; q < field_ids_.size(); ++q) {
        int field_id = field_ids_[q];
        const string &s = item.file_fields.at(field_id);
        Expects(s.size() == field_sizes[q]);
        batch_data->file_fields[field_id].append(s);
      }
    }

    num_examples_dequeued_ += data.size();

    {
      while (true) {
        std::unique_lock<std::mutex> lock(batch_lock_);
        bool ok = batch_cv_.wait_for(lock, std::chrono::milliseconds(100), [&] { return batch_data_ == nullptr; });
        if (ok) {
          Expects(batch_data_ == nullptr);
          batch_data_ = std::move(batch_data);
          break;
        } else { // timeout.
          if (stop_requested_ or queue_->is_closed_and_empty()) {
            goto BatchRoutine_END;
          }
        }
      }
    }

    batch_cv_.notify_one();
  }

  BatchRoutine_END:

  LOG(INFO) << "End of BatchRoutine() after reading " << num_examples_dequeued_.load() << " examples.";
}

void BatchLoader::StartWorkers() {
  std::lock_guard<std::mutex> lock(lock_);

  if (num_active_threads_ > 0 || queue_ != nullptr) {
    LOG(FATAL) << "Threads are already running: " << num_active_threads_;
    return;
  }

  stop_requested_ = false;
  queue_ = make_unique<mvshape::concurrency::BatchQueue<BatchData>>(batch_size_ * kQueueSizeMultiplier);

  num_active_threads_ = kNumReaderThreads + 1;

  batch_thread_ = std::thread([&] {
    BatchRoutine(0);
    {
      std::lock_guard<std::mutex> thread_lock(lock_);
      --num_active_threads_;
    }
    cv_.notify_all();
  });

  for (int i = 0; i < kNumReaderThreads; ++i) {
    reader_threads_.emplace_back([&] {
      DataReaderRoutine(i);
      {
        std::lock_guard<std::mutex> thread_lock(lock_);
        --num_active_threads_;
      }
      cv_.notify_all();
    });
  }

  std::stringstream stream;
  stream << "Launched " << kNumReaderThreads + 1 << " threads fetching " << size() << " examples";
  if (is_seamless_) {
    stream << " indefinitely.";
  } else {
    stream << " once.";
  }
  LOG(INFO) << stream.str();
}

void BatchLoader::StopWorkers() {
  stop_requested_ = true;
  queue_->Close();
  cv_.notify_all();
  batch_cv_.notify_all();

  auto join_start = mvshape::MicroSecondsSinceEpoch();
  for (int j = 0; j < reader_threads_.size(); ++j) {
    DLOG(INFO) << "Joining reader thread " << j;
    reader_threads_[j].join();
  }
  DLOG(INFO) << "Joining batch thread";
  batch_thread_.join();

  reader_threads_.clear();

  {
    std::lock_guard<std::mutex> lock(lock_);
    queue_ = nullptr;
  }
  cv_.notify_all();
  batch_cv_.notify_all();

  LOG(INFO) << "Joined all threads. " << "Time elapsed: "
            << (mvshape::MicroSecondsSinceEpoch() - join_start) << " microseconds.";
}

void BatchLoader::StopWorkersAsync() {
  DLOG(INFO) << "StopWorkersAsync()";
  auto start_time = mvshape::MicroSecondsSinceEpoch();

  stop_requested_ = true;
  queue_->Close();

  auto task = std::thread([=] {

    while (true) {
      bool no_timeout;
      {
        std::unique_lock<std::mutex> lock(lock_);
        no_timeout = cv_.wait_for(lock, std::chrono::seconds(kThreadJoinWaitSeconds), [&] {
          return num_active_threads_ <= 0;
        });
      }

      if (no_timeout) {
        auto join_start = mvshape::MicroSecondsSinceEpoch();
        for (int j = 0; j < reader_threads_.size(); ++j) {
          reader_threads_[j].join();
        }
        batch_thread_.join();
        // Sanity check.
        Ensures(mvshape::MicroSecondsSinceEpoch() - join_start < 1e6);

        auto end_time = mvshape::MicroSecondsSinceEpoch();
        LOG(INFO) << "All threads stopped. Time elapsed: " << end_time - start_time << " microseconds";

        reader_threads_.clear();
        {
          std::lock_guard<std::mutex> lock(lock_);
          queue_ = nullptr;
        }
        cv_.notify_all();
        break;
      } else {
        LOG(WARNING) << num_active_threads_ << " threads are still running.";
      }

    }
  });
  task.detach();
}

std::unique_ptr<BatchData> BatchLoader::Next() {
  if (stop_requested_) {
    LOG(FATAL) << "Next() called after requesting stop.";
  }

  auto start = MicroSecondsSinceEpoch();
  std::unique_ptr<BatchData> ret;

  if (!is_seamless_ && num_examples_returned_ >= size()) {
    LOG(INFO) << "End of queue. Returned " << num_examples_returned_ << " examples.";
    ret = nullptr;
  } else if (queue_ == nullptr) {
    LOG(INFO) << "Queue is inactive. Number of examples returned: " << num_examples_returned_;
    ret = nullptr;
  } else {
    {
      std::unique_lock<std::mutex> lock(batch_lock_);
      batch_cv_.wait(lock, [&] { return batch_data_ != nullptr; });

      Expects(batch_data_ != nullptr);
      ret = std::move(batch_data_);
      Ensures(batch_data_ == nullptr);
    }

    num_examples_returned_ += ret->size;
    batch_cv_.notify_one();
  }

  auto elapsed = MicroSecondsSinceEpoch() - start;
  if (elapsed > kSlowIOWarningMicroSec) {
    LOG(WARNING) << "IO bottleneck detected. Waiting time: "
                 << std::fixed << std::setprecision(2)
                 << elapsed / 1e3 << " ms.";
  }

  return ret;
}

int BatchLoader::num_examples_returned() const {
  return num_examples_returned_;
}

}
}
