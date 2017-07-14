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
#include "cpp/lib/mvshape_helpers.h"
#include "proto/dataset.pb.h"

#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

using namespace mvshape;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  Expects(!FLAGS_run_id.empty());

  tf::SavedModelBundle model;

  int global_step = mvshape::tf_utils::LoadSavedModel({"mv"}, &model);
  Expects(0 == global_step);
  tf::graph::SetDefaultDevice(FLAGS_default_device, model.meta_graph_def.mutable_graph_def());

  tf::Session *session = model.session.get();

  // Continue from a previously saved checkpoint, if one exists.
  try {
    auto last_checkpoint = tf_utils::FindLastCheckpoint();
    tf_utils::RestoreCheckpoint(session, last_checkpoint);
  } catch (const std::runtime_error &err) {
  }


  //////////////////////////////////////////

//  auto experiment = mv::VIEWER_CENTERED;
  auto experiment = mv::OBJECT_CENTERED;

  mv::Examples train_examples;
  mv::Examples test_examples;

  switch (experiment) {
    case mv::VIEWER_CENTERED:
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &train_examples);
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/validation.bin"), &train_examples);

      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &test_examples);
      break;

    case mv::OBJECT_CENTERED:
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/train.bin"), &train_examples);
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/validation.bin"), &train_examples);

      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/test.bin"), &test_examples);
      break;

    default:
      throw std::runtime_error("not implemented");
  }

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

//    timer.Toc();

    vector<tf::Tensor> out;
    TF_CHECK_OK(model.session->Run(feed, {"iou"}, {"train_op"}, &out));

    timer.Toc();

    int num_batches_read = loader.num_examples_returned() / batch_size;
    if (num_batches_read % 10 == 0) {
      LOG(INFO) << loader.epoch() << ", " << loader.num_examples_returned_in_current_epoch()
                << "/" << loader.size() << ", iou: " << out[0].scalar<float>()() << ", "
                << timer.Duration() << " s per batch. "
                << timer.Duration() / static_cast<float>(batch_size) << " s per example. "
                << timer.Duration() / static_cast<float>(batch_size) * loader.size() << " s per epoch. "
                << loader.num_examples_returned() << " examples total.";
    }

    if (tf_utils::DidChange(loader.epoch())) {
      tf_utils::IncrementEpochCount(session);
      mvshape::evaluation::Shrec12(session, test_examples);
      tf_utils::SaveCheckpoint(session);
    }
  }

  loader.StopWorkers();
  model.session->Close();

}
