//
// Created by daeyun on 7/12/17.
//

#include "tf_utils.h"

#include <stdexcept>
#include <gsl/gsl_assert>
#include <boost/filesystem.hpp>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

#include "flags.h"
#include "file_io.h"
#include "string_utils.h"

namespace mvshape {
namespace tf_utils {
namespace fs = boost::filesystem;

int LoadSavedModel(const std::unordered_set<std::string> &tags, tf::SavedModelBundle *model) {
  auto session_options = tf::SessionOptions();
  auto run_options = tf::RunOptions();

  if (FLAGS_is_test_mode) {
    session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.02);
    session_options.config.mutable_gpu_options()->set_allow_growth(true);

    run_options.set_timeout_in_ms(1000);
    run_options.set_trace_level(tf::RunOptions_TraceLevel_FULL_TRACE);

    LOG(INFO) << "Initializing session using test mode config.";
  }

  session_options.config.mutable_gpu_options()->set_allow_growth(true);

  if (FLAGS_tf_model.empty()) {
    LOG(ERROR) << "--tf_model flag missing.";
    throw std::runtime_error("");
  }

  if (!fs::is_directory(FLAGS_tf_model)) {
    LOG(ERROR) << "--tf_model must be a directory. " << FLAGS_tf_model;
    throw std::runtime_error("");
  }

  auto saved_model_path = fs::canonical(FLAGS_tf_model);
  if (!fs::is_regular_file(saved_model_path / "saved_model.pb")
      and !fs::is_regular_file(saved_model_path / "saved_model.pbtxt")) {
    LOG(ERROR) << "Invalid --tf_model. " << FLAGS_tf_model;
    throw std::runtime_error("");
  }

  LOG(INFO) << "Loading saved model. " << saved_model_path.string();

  auto status = tf::LoadSavedModel(session_options, run_options, saved_model_path.string(), tags, model);
  TF_CHECK_OK(status);

  auto step_epoch = ScalarOutput<int>(model->session.get(), vector<string>{"global_step", "epoch"});

  LOG(INFO) << "Global step " << step_epoch[0] << ", epoch: " << step_epoch[1];

  return step_epoch[0];
}

tf::TensorShape VectorAsTensorShape(const vector<tf::int64> &shape) {
  return tf::TensorShape(tf::gtl::ArraySlice<tf::int64>(shape));
}

vector<int> TensorShapeAsVector(const tf::TensorShape &shape) {
  vector<int> ret;
  for (int i = 0; i < shape.dims(); ++i) {
    ret.push_back(shape.dim_size(i));
  }
  return ret;
}

string SaveCheckpoint(tf::Session *session) {
  tf::Tensor filename_tensor(tf::DT_STRING, {1, 1});

  vector<int> epoch_step = ScalarOutput<int>(session, vector<string>{"epoch", "global_step"});

  auto name = WithLeadingZeros(epoch_step[0], 4) + "_" + WithLeadingZeros(epoch_step[1], 9);

  string tf_model_dir = FLAGS_tf_model;
  if (tf_model_dir.back() == '/') {
    tf_model_dir = tf_model_dir.substr(0, tf_model_dir.size() - 1);
  }

  Expects(!FLAGS_run_id.empty());

  auto out_dir = fs::path(FLAGS_out_dir) / "tf_out" / fs::basename(tf_model_dir) / FLAGS_run_id / "checkpoints";

  if (!fs::exists(out_dir)) {
    fs::create_directories(out_dir);
    LOG(INFO) << "mkdir -p " << out_dir.string();
  }

  string out_file = (out_dir / name).string();

  filename_tensor.matrix<std::string>()(0, 0) = out_file;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
      {"saver/Const:0", filename_tensor},  // Filename
  };
  LOG(INFO) << "Saving " << out_file;

  // https://stackoverflow.com/a/37671613
  TF_CHECK_OK(session->Run(feed, {}, {"saver/control_dependency"}, nullptr));

  return out_file;
};

int RestoreCheckpoint(tf::Session *session, const string &filename) {
  tf::Tensor filename_tensor(tf::DT_STRING, {1, 1});

  filename_tensor.matrix<std::string>()(0, 0) = filename;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
      {"saver/Const:0", filename_tensor},  // Filename
  };

  TF_CHECK_OK(session->Run(feed, {}, {"saver/restore_all"}, nullptr));

  auto step_epoch = ScalarOutput<int>(session, vector<string>{"global_step", "epoch"});

  LOG(INFO) << "Restored checkpoint. Global step " << step_epoch[0] << ", epoch: " << step_epoch[1] << " in "
            << filename;

  return step_epoch[0];
};

int IncrementEpochCount(tf::Session *session) {
  TF_CHECK_OK(session->Run({}, {}, {"increment_epoch"}, nullptr));
  return ScalarOutput<int>(session, "epoch");
}

string SaveTensor(tf::Session *session, const tf::Tensor &tensor, const string &name) {
  Expects(!name.empty());

  vector<int> epoch_step = ScalarOutput<int>(session, vector<string>{"epoch", "global_step"});
  auto step_name = WithLeadingZeros(epoch_step[0], 4) + "_" + WithLeadingZeros(epoch_step[1], 9);

  string tf_model_dir = FLAGS_tf_model;
  if (tf_model_dir.back() == '/') {
    tf_model_dir = tf_model_dir.substr(0, tf_model_dir.size() - 1);
  }
  auto out_dir = mvshape::FileIO::FullOutPath((fs::path("tf_out") / fs::basename(tf_model_dir) /
      FLAGS_run_id / "tensors" / step_name).string());

  if (!fs::exists(out_dir)) {
    fs::create_directories(out_dir);
    LOG(INFO) << "mkdir -p " << out_dir;
  }

  string out_file = (fs::path(out_dir) / (name + ".bin")).string();

  LOG(INFO) << "Saving " << out_file;

  mvshape::FileIO::SerializeTensor<float>(
      out_file, tensor.flat<float>().data(), mvshape::tf_utils::TensorShapeAsVector(tensor.shape()));

  return out_file;
}

}
}
