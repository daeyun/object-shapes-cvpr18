//
// Created by daeyun on 7/13/17.
//

#include "mvshape_helpers.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/graph/default_device.h"

#include "cpp/lib/benchmark.h"
#include "cpp/lib/database.h"
#include "cpp/lib/file_io.h"
#include "cpp/lib/data_io.h"
#include "cpp/lib/tf_utils.h"
#include "cpp/lib/flags.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace mvshape {
namespace evaluation {

std::map<string, float> Shrec12(tf::Session *session, const mv::Examples &eval_examples) {
  mvshape::Timer timer("eval");
  std::map<string, float> results;

  vector<int> step_epoch = tf_utils::ScalarOutput<int>(session, vector<string>{"global_step", "epoch"});
  int epoch = step_epoch[1];
  int global_step = step_epoch[0];
  int batch_size = FLAGS_batch_size;

  LOG(INFO) << "== Starting evaluation. epoch: " << epoch << ", global step: " << global_step;

  std::map<int, mv::Examples> eval_examples_by_tag = tf_utils::SplitExamplesByTags(
      eval_examples, {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS});

  // Sanity check.
  if (eval_examples.split_name() == mv::TEST) {
    // Specific to shrec12.
    Expects(600 == eval_examples_by_tag[mv::NOVELVIEW].examples_size());
    Expects(600 == eval_examples_by_tag[mv::NOVELMODEL].examples_size());
    Expects(600 == eval_examples_by_tag[mv::NOVELCLASS].examples_size());
  } else {
    throw std::runtime_error("not implemented");
  }

  auto ph = Shrec12Placeholders::Build(batch_size);
  ph.is_training.scalar<bool>()() = false;

  const vector<string> float_scalar_names{"loss", "iou"};

  for (int tag: {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS}) {
    mvshape::Data::BatchLoader eval_loader(&eval_examples_by_tag[tag], {
        mv::Example::kSingleDepthFieldNumber,
        mv::Example::kMultiviewDepthFieldNumber,
    }, batch_size, false);

    int example_count = 0;

    while (true) {
      auto batch = eval_loader.Next();

      if (batch == nullptr) {
        // End of queue.
        Expects(example_count == eval_loader.size());
        break;
      }

      tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kSingleDepthFieldNumber), &ph.in_depth);
      tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kMultiviewDepthFieldNumber), &ph.target_depth);

      vector<pair<string, tf::Tensor>> feed = {
          {"placeholder/is_training", ph.is_training},
          {"placeholder/target_depth_offset", ph.target_depth_offset},
          {"placeholder/in_depth", ph.in_depth},
          {"placeholder/target_depth", ph.target_depth},
      };

      auto result = tf_utils::ScalarOutput<float>(session, float_scalar_names, feed);
      for (int i = 0; i < float_scalar_names.size(); ++i) {
        const auto &name = float_scalar_names[i];
        auto it = results.find(name);
        if (it == results.end()) {
          results[name] = 0;
        }
        results[name] += result[i] * batch->size;
      }

      example_count += batch->size;
    }
    for (int i = 0; i < float_scalar_names.size(); ++i) {
      results[float_scalar_names[i]] /= static_cast<float>(example_count);
    }

    std::stringstream stream;
    stream << mv::Tag_Name(static_cast<mv::Tag>(tag)) << ". ";

    for (int i = 0; i < float_scalar_names.size(); ++i) {
      stream << float_scalar_names[i] << ": " << results[float_scalar_names[i]] << ",  ";
    }

    LOG(INFO) << "EVAL  " << stream.str();

    eval_loader.StopWorkers();
  }

  timer.Toc();
  LOG(INFO) << "Evaluation took " << timer.Duration() << " seconds.";

  return results;
}

}
}
