//
// Created by daeyun on 7/12/17.
//

#include <mutex>
#include <condition_variable>
#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <concurrentqueue/blockingconcurrentqueue.h>

#pragma once

namespace mvshape {
namespace concurrency {

static constexpr int64_t kTimeoutMicroSec = 1500;

template<typename T>
class BatchQueue {
 public:
  // Non-assignable.
  BatchQueue(const BatchQueue &) = delete;
  BatchQueue &operator=(BatchQueue const &) = delete;

  explicit BatchQueue(size_t max_size) : queue_(),
                                         max_size_(max_size) {}

  bool Enqueue(const T &item) {
    {
      std::unique_lock<std::mutex> lock(lock_);
      bool can_enqueue = size_ < max_size_ && !is_closed_;
      while (!can_enqueue) {
        can_enqueue = cv_.wait_for(lock, std::chrono::microseconds(kTimeoutMicroSec),
                                   [&] { return size_ < max_size_; });
        if (is_closed_) {
          return false;
        }
      }
      size_++;
    }

    queue_.enqueue(item);
    return true;
  }

  template<typename It>
  size_t Dequeue(It first, size_t num) {
    size_t count = 0;

#ifndef NDEBUG
    It iterator_copy = first;
#endif

    while (count < num) {
      // Advances the iterator.
      size_t num_dequeued = queue_.template wait_dequeue_bulk_timed<It &>(first, num - count, kTimeoutMicroSec);

      count += num_dequeued;

      if (num_dequeued > 0) {
        {
          std::lock_guard<std::mutex> lock(lock_);
          size_ -= num_dequeued;
        }
        cv_.notify_all();
      } else if (is_closed_) {
        Expects(size_ == 0);
        break;
      }

#ifndef NDEBUG
      Ensures(size_ >= 0);
      Ensures(std::distance(iterator_copy, first) == count);
#endif
    }

    Expects(count == num || (is_closed_ && count <= num));

    return count;
  }

  void Close() {
    is_closed_ = true;
  }

  // Closed means no more enqueues will happen. Dequeues can happen.
  bool is_closed() const { return is_closed_; }

 private:
  BatchQueue() {}

  moodycamel::BlockingConcurrentQueue <T> queue_;
  size_t max_size_;
  volatile bool is_closed_ = false;
  int size_ = 0;
  std::mutex lock_;
  std::condition_variable cv_;
};

}
}

