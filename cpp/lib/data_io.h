//
// Created by daeyun on 6/29/17.
//
#pragma once

#include <vector>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <concurrentqueue/blockingconcurrentqueue.h>
#include "common.h"
#include "camera.h"
#include "proto/dataset.pb.h"

namespace mvshape {
namespace Data {

namespace mv = mvshape_dataset;
using moodycamel::BlockingConcurrentQueue;

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
              bool shuffle);

  void StartWorkers();
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

  int num_examples_returned() const;

  static constexpr int kNumReaderThreads = 4;
  static constexpr int kQueueCapacityMultiplier = 3;
  static constexpr int kSlowIOWarningMicroSec = 10000;

 private:

  void DataReaderRoutine(int thread_id);
  void BatchRoutine(int thread_id);

  RenderingReader reader_;
  const vector<int> field_ids_;
  int batch_size_;
  bool is_seamless_;
  bool shuffle_;
  BlockingConcurrentQueue<BatchData> queue_;
  vector<int> indices_;
  int current_read_index_ = 0;
  int num_examples_fetched_ = 0;
  int num_examples_returned_ = 0;
  std::mutex index_lock_;
  std::mutex queue_lock_;
  std::mutex batch_lock_;
  volatile bool should_stop_workers_ = false;
  vector<std::thread> reader_threads_;
  std::thread batch_thread_;
  std::condition_variable cv_;
  std::condition_variable batch_cv_;
  std::unique_ptr<BatchData> batch_data_ = nullptr;
  bool end_of_queue_ = false;
};

}
}
