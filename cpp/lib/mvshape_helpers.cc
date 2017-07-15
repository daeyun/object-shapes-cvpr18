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
#include "string_utils.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace mvshape {
namespace evaluation {

std::map<string, float> Shrec12(tf::Session *session, const mv::Examples &eval_examples, bool save_tensors) {
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

  const vector<string> float_scalar_names{"loss", "iou", "loss_s", "loss_d"};
  const vector<string> float_tensor_names{"out_depth", "out_silhouette"};

  for (int tag: {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS}) {

    int num_workers = (save_tensors) ? 1 : Data::kNumReaderThreads;

    mvshape::Data::BatchLoader eval_loader(&eval_examples_by_tag[tag], {
        mv::Example::kSingleDepthFieldNumber,
        mv::Example::kMultiviewDepthFieldNumber,
    }, batch_size, false, num_workers);

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

      if (save_tensors) {
        auto i_batch = eval_loader.num_batches_returned();

        // Saving output tensors
        vector<tf::Tensor> out;
        TF_CHECK_OK(session->Run(feed, float_tensor_names, {}, &out));

        Ensures(out.size() == float_tensor_names.size());
        for (int j = 0; j < out.size(); ++j) {
          string out_name = (fs::path(mv::Tag_Name(static_cast<mv::Tag>(tag))) / float_tensor_names[j]
              / std::to_string(i_batch - 1)).string();
          tf_utils::SaveTensor(session, out[j], out_name);
        }

        // Saving input tensors
        for (const auto &item: feed) {
          string placeholder_name = item.first;
          std::replace(placeholder_name.begin(), placeholder_name.end(), '/', '_');
          const auto &tensor = item.second;
          if (tensor.dims() == 0) {
            LOG(INFO) << "Skipping scalar " << item.first;
            continue;
          }
          string out_name = (fs::path(mv::Tag_Name(static_cast<mv::Tag>(tag))) / placeholder_name
              / std::to_string(i_batch - 1)).string();
          tf_utils::SaveTensor(session, tensor, out_name);
        }

        // Saving indices because BatchLoader shuffles.
        string out_name = (fs::path(mv::Tag_Name(static_cast<mv::Tag>(tag))) / "index"
            / std::to_string(i_batch - 1)).string();
        FileIO::SerializeTensor<int>(tf_utils::FindAndPrepareOutputDirectory(session, out_name),
                                     batch->example_indices.data(),
                                     vector<int>{batch->example_indices.size()});
      }  // save_tensors





    }
    for (int i = 0; i < float_scalar_names.size(); ++i) {
      results[float_scalar_names[i]] /= static_cast<float>(example_count);
    }

    std::stringstream stream;
    stream << mv::Tag_Name(static_cast<mv::Tag>(tag)) << ". ";

    for (int i = 0; i < float_scalar_names.size(); ++i) {
      stream << float_scalar_names[i] << ": " << results[float_scalar_names[i]] << ",  ";
    }

    LOG(INFO) << "#############################################";
    LOG(INFO) << "EVAL  " << stream.str();
    LOG(INFO) << "#############################################";

    eval_loader.StopWorkers();
  }

  timer.Toc();
  LOG(INFO) << "Evaluation took " << timer.Duration() << " seconds.";

  return results;
}

}
}
