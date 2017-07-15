//
// Created by daeyun on 6/29/17.
//
#pragma once

#include <vector>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "common.h"
#include "camera.h"
#include "batch_queue.h"
#include "proto/dataset.pb.h"

namespace mvshape {
namespace Data {

constexpr float kNear = 0.1;
constexpr float kFar = 40;
constexpr int kImageNormalizationUpScale = 4;
constexpr int kNormalizationPadding = 1;

constexpr int kNumReaderThreads = 4;
constexpr int kQueueSizeMultiplier = 3;
constexpr int kSlowIOWarningMicroSec = 10000;
constexpr int kThreadJoinWaitSeconds = 5;

namespace mv = mvshape_dataset;

void GenerateDataset();

int SaveExamples(const string &sqlite_file, const string &view_name, const string &out_dir);

int LoadExamples(const string &filename, mv::Examples *examples);

int LoadExamples(const vector<string> &filenames, mv::Examples *examples);

// data.data() is the pointer to the raw data, in row-major order.
void LoadRenderingContent(const mv::Rendering &rendering, string *data);
void LoadRenderingContent(const string &full_filename, string *data);

void SixViews(const Camera &camera, vector<unique_ptr<Camera>> *out);

class RenderingReader {
 public:
  explicit RenderingReader(const mv::Examples *examples) : examples_(examples) {
  }

  void Read(int row_index,
            const vector<int> &field_ids,
            std::unordered_map<int, unique_ptr<string>> *field_data);

  vector<int> LoadBatch(const vector<int> &field_ids,
                        int batch_size,
                        std::unordered_map<int, vector<unique_ptr<string>>> *field_data);

  const mvshape_dataset::Examples *examples() const {
    return examples_;
  }

 private:
  void PrepareIndices();
  void RandomIndices(int n, vector<int> *indices);

  const mv::Examples *examples_;
  vector<int> indices_;
};

struct BatchData {
  // TODO: rename file_fields.
  std::unordered_map<int, string> file_fields;
  std::vector<int> example_indices;
  int size = 0;
};

class BatchLoader {
 public:
  BatchLoader(const mv::Examples *examples,
              const vector<int> &field_ids,
              int batch_size,
              bool is_seamless,
              int num_workers=kNumReaderThreads);

  // Deprecated.
  void StopWorkersAsync();

  void StopWorkers();

  std::unique_ptr<BatchData> Next();

  int size() const {
    return reader_.examples()->examples_size();
  }

  int epoch() const {
    return num_examples_returned() / size();
  };

  int num_examples_returned_in_current_epoch() const {
    return num_examples_returned() % size();
  };

  int num_batches_returned() const {
    return num_examples_returned() / batch_size_;
  };

  int num_examples_returned() const;

 private:
  void StartWorkers();

  void DataReaderRoutine(int thread_id);
  void BatchRoutine(int thread_id);

  RenderingReader reader_;
  const vector<int> field_ids_;
  const int batch_size_;
  bool is_seamless_;
  unique_ptr<mvshape::concurrency::BatchQueue<BatchData>> queue_;
  vector<int> indices_;
  int current_read_index_ = 0;
  std::atomic<int> num_examples_dequeued_;
  std::atomic<int> num_examples_enqueued_;
  int num_examples_returned_ = 0;
  int num_active_threads_ = 0;
  bool stop_requested_ = false;
  int num_workers_;

  std::mutex lock_;
  std::mutex batch_lock_;
  vector<std::thread> reader_threads_;
  std::thread batch_thread_;
  std::condition_variable cv_;
  std::condition_variable batch_cv_;
  std::unique_ptr<BatchData> batch_data_ = nullptr;
};

}
}
