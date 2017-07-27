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
#include "cpp/lib/string_utils.h"

#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>

DECLARE_string(tf_model);
DECLARE_string(run_id);
DECLARE_int32(batch_size);
DECLARE_string(default_device);
DECLARE_int32(restore_epoch);

using namespace mvshape;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  Expects(!FLAGS_run_id.empty());

  tf::SavedModelBundle model;

  int global_step = mvshape::tf_utils::LoadSavedModel({"mv"}, &model);
  Expects(0 == global_step);
  tf::graph::SetDefaultDevice(FLAGS_default_device, model.meta_graph_def.mutable_graph_def());

  tf::Session *session = model.session.get();

  // Continue from a previously saved checkpoint, if one exists.
  string checkpoint;
  if (FLAGS_restore_epoch < 0) {
    checkpoint = tf_utils::FindLastCheckpoint();
  } else {
    LOG(INFO) << "--restore_epoch=" << FLAGS_restore_epoch;
    checkpoint = tf_utils::FindCheckpointAtEpoch(FLAGS_restore_epoch);
  }

  if (!checkpoint.empty()) {
    tf_utils::RestoreCheckpoint(session, checkpoint);
    vector<int> epoch_step = tf_utils::ScalarOutput<int>(session, vector<string>{"epoch", "global_step"});
    if (FLAGS_restore_epoch >= 0) {
      Ensures(FLAGS_restore_epoch == epoch_step[0]);
    }
  }


  //////////////////////////////////////////

  string run_id_lower = mvshape::ToLower(FLAGS_run_id);
  LOG(INFO) << "run_id_lower: " << run_id_lower;

  mv::Tag experiment = mv::VIEWER_CENTERED;

  if (run_id_lower.find("viewer_centered") != string::npos) {
    experiment = mv::VIEWER_CENTERED;
  } else if (run_id_lower.find("object_centered") != string::npos) {
    experiment = mv::OBJECT_CENTERED;
  } else {
    throw std::runtime_error("not implemented");
  }

  LOG(INFO) << "##################";
  LOG(INFO) << "experiment: " << mv::Tag_Name(experiment);
  LOG(INFO) << "##################";

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
      mvshape::evaluation::Shrec12(session, test_examples, false);
      tf_utils::SaveCheckpoint(session);
    }
  }

  loader.StopWorkers();
  model.session->Close();

}
