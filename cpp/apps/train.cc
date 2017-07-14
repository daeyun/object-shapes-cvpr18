//
// Created by daeyun on 7/4/17.
//
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
#include "proto/dataset.pb.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

namespace tf = tensorflow;
namespace mv = mvshape_dataset;
namespace fs = boost::filesystem;

using namespace mvshape;

DEFINE_int32(batch_size, 50, "Number of batches per training step.");
DEFINE_string(default_device, "/gpu:0", "TensorFlow device.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  Expects(!FLAGS_run_id.empty());

  tf::SavedModelBundle model;

  int global_step = mvshape::tf_utils::LoadSavedModel({"mv"}, &model);
  tf::graph::SetDefaultDevice(FLAGS_default_device, model.meta_graph_def.mutable_graph_def());

  Expects(0 == global_step);

  tf::Session* session = model.session.get();



  // Continue from previously saved checkpoint, if one exists.
  try {
    auto last_checkpoint = tf_utils::FindLastCheckpoint();
    tf_utils::RestoreCheckpoint(session, last_checkpoint);
  } catch (const std::runtime_error &err) {
  }





  //////////////////////////////////////////
  mvshape_dataset::Examples train_examples;
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &train_examples);
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/validation.bin"),
                              &train_examples);

  mvshape_dataset::Examples test_examples;
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &test_examples);

  std::map<int, mv::Examples> test_examples_by_tag = tf_utils::SplitExamplesByTags(
      test_examples, {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS});

  // Specific to shrec12.
  Expects(600 == test_examples_by_tag[mv::NOVELVIEW].examples_size());
  Expects(600 == test_examples_by_tag[mv::NOVELMODEL].examples_size());
  Expects(600 == test_examples_by_tag[mv::NOVELCLASS].examples_size());

  int batch_size = FLAGS_batch_size;
  //////////////////////////////////////////












  // Placeholder tensors.

  tf::Tensor target_depth_offset = tf_utils::MakeScalarTensor<tf::DT_FLOAT>(-5.5);
  tf::Tensor is_training = tf_utils::MakeScalarTensor<tf::DT_BOOL>(true);

  tf::Tensor in_depth(tf::DT_FLOAT, {batch_size, 128, 128, 1});
  tf::Tensor target_depth(tf::DT_FLOAT, {batch_size, 6, 128, 128, 1});

  //////////////////////////////////////////

  mvshape::Data::BatchLoader loader(&train_examples, {
      mv::Example::kSingleDepthFieldNumber,
      mv::Example::kMultiviewDepthFieldNumber,
  }, batch_size, true);

  mvshape::Timer timer("train");

  while (true) {
    // Returns immediately, unless there's an IO bottleneck.
    auto batch = loader.Next();

    // TODO
    if (batch == nullptr) {
      break;
    }

    timer.Tic();

    is_training.scalar<bool>()() = true;
    tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kSingleDepthFieldNumber), &in_depth);
    tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kMultiviewDepthFieldNumber), &target_depth);

    vector<pair<string, tf::Tensor>> feed = {
        {"placeholder/is_training", is_training},
        {"placeholder/target_depth_offset", target_depth_offset},
        {"placeholder/in_depth", in_depth},
        {"placeholder/target_depth", target_depth},
    };

    timer.Toc();

    vector<tf::Tensor> out;
    TF_CHECK_OK(model.session->Run(feed, {"iou"}, {"train_op"}, &out));

//    timer.Toc();

    int num_batches_read = loader.num_examples_returned() / batch_size;
    if (num_batches_read % 10 == 0) {
      LOG(INFO) << loader.epoch() << ", " << loader.num_examples_returned_in_current_epoch()
                << "/" << loader.size() << ", iou: " << out[0].scalar<float>()() << ", "
                << timer.Duration() << " s per batch. "
                << timer.Duration() / static_cast<float>(batch_size) << " s per example. "
                << timer.Duration() / static_cast<float>(batch_size) * loader.size() << " s per epoch. "
                << loader.num_examples_returned() << " examples total.";
    }

    // TODO
    if (tf_utils::DidChange(loader.epoch())) {
      tf_utils::IncrementEpochCount(session);

      LOG(INFO) << "Starting evaluation at " << loader.epoch();

      mvshape::Timer eval_timer("eval");

      for (int tag: {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS}) {
        mvshape::Data::BatchLoader eval_loader(&test_examples_by_tag[tag], {
            mv::Example::kSingleDepthFieldNumber,
            mv::Example::kMultiviewDepthFieldNumber,
        }, batch_size, false);

        int count = 0;
        std::map<string, float> mean;

        vector<string> names{"loss", "iou"};
        while (true) {
          auto b = eval_loader.Next();

          if (b == nullptr) {
            Expects(count == eval_loader.size());
            break;
          }

          is_training.scalar<bool>()() = false;
          tf_utils::SetTensorData<float>(b->file_fields.at(mv::Example::kSingleDepthFieldNumber), &in_depth);
          tf_utils::SetTensorData<float>(b->file_fields.at(mv::Example::kMultiviewDepthFieldNumber), &target_depth);

          auto result = tf_utils::ScalarOutput<float>(session, names, feed);
          for (int i = 0; i < names.size(); ++i) {
            const auto &name = names[i];
            auto it = mean.find(name);
            if (it == mean.end()) {
              mean[name] = 0;
            }
            mean[name] += result[i] * b->size;
          }

          count += b->size;
        }
        for (int i = 0; i < names.size(); ++i) {
          mean[names[i]] /= static_cast<float>(count);
        }

        std::stringstream stream;
        stream << mv::Tag_Name(static_cast<mv::Tag>(tag)) << ". ";

        for (int i = 0; i < names.size(); ++i) {
          stream << names[i] << ": " << mean[names[i]] << ",  ";
        }

        LOG(INFO) << stream.str();

        eval_loader.StopWorkers();
      }

      tf_utils::SaveCheckpoint(session);

      eval_timer.Toc();

      vector<int> step_epoch = tf_utils::ScalarOutput<int>(session, vector<string>{"global_step", "epoch"});

      LOG(INFO) << "=== End of epoch " << step_epoch[1] << ". Global step: " << step_epoch[0] << " ===";
      LOG(INFO) << "Evaluation and saving took " << eval_timer.Duration() << " seconds.";

    }  // end of DidChange(epoch)

  }

  loader.StopWorkers();
  model.session->Close();

}
