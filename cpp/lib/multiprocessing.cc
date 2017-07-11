//
// Created by daeyun on 6/30/17.
//

#include "multiprocessing.h"

#include <sys/wait.h>

#include <glog/logging.h>
#include <gsl/gsl_assert>

namespace mvshape {
namespace MP {

vector<pair<int, int>> SplitRange(int end, int n) {
  Expects(end >= 0);
  Expects(n > 0);
  const int div = end / n;
  const int rem = end % n;

  vector<pair<int, int>> indices;

  int start = 0;
  for (int i = 0; i < n; ++i) {
    int chunk_size;
    if (i < rem) {
      chunk_size = div + 1;
    } else {
      chunk_size = div;
    }

    indices.emplace_back(start, start + chunk_size);
    start += chunk_size;
  }

  Ensures(start == end);
  return indices;
}

int MultiFork(int work_size, int num_processes, pair<int, int> *start_end_indices) {
  Expects(num_processes > 0);
  N = num_processes;
  auto ranges = SplitRange(work_size, num_processes);

  int pid = 0;
  for (int j = 0; j < num_processes - 1; ++j) {
    auto s = fork();
    if (s < 0) {
      LOG(ERROR) << "Error in fork(). errno: " << errno;
      exit(EXIT_FAILURE);
    } else if (s == 0) {
      // Child.
      pid = j + 1;
      break;
    }
    // Parent.
    LOG(INFO) << "PID " << s << " started.";
  }

  *start_end_indices = ranges[pid];

  process_num = pid;
  return pid;
}

void Join() {
  if (process_num == 0) {
    int status;
    pid_t pid;
    for (int i = 0; i < N - 1; ++i) {
      pid = wait(&status);
      LOG(INFO) << "PID " << pid << " exited. (" << i + 1 << " out of " << N - 1 << ", status " << status << ")";
    }
  } else {
    _Exit(EXIT_SUCCESS);
  }
}

}
}
