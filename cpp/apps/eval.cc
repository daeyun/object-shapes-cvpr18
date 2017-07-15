//
// Created by daeyun on 7/4/17.
//
#include <algorithm>
#include <string>

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
#include "cpp/lib/string_utils.h"
#include "proto/dataset.pb.h"

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
  gflags::ParseCommandLineFlags(&argc, &argv, true);
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

  LOG(INFO) << "experiment: " << mv::Tag_Name(experiment);

//  mv::Examples train_examples;
  mv::Examples test_examples;

  switch (experiment) {
    case mv::VIEWER_CENTERED:
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &test_examples);
      break;

    case mv::OBJECT_CENTERED:
      Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/test.bin"), &test_examples);
      break;

    default:
      throw std::runtime_error("not implemented");
  }

  int batch_size = FLAGS_batch_size;
  //////////////////////////////////////////


  mvshape::evaluation::Shrec12(session, test_examples, true);

  model.session->Close();

}
