//
// Created by daeyun on 7/12/17.
//

#include "tf_utils.h"

#include <unordered_set>
#include <stdexcept>
#include <gsl/gsl_assert>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string.hpp>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

#include "flags.h"
#include "file_io.h"
#include "string_utils.h"
#include "proto/dataset.pb.h"

namespace mvshape {
namespace tf_utils {
namespace fs = boost::filesystem;
namespace mv = mvshape_dataset;

int LoadSavedModel(const std::unordered_set<std::string> &tags, tf::SavedModelBundle *model) {
  auto session_options = tf::SessionOptions();
  auto run_options = tf::RunOptions();

//  session_options.config.mutable_gpu_options()->set_allow_growth(true);
//  session_options.config.mutable_gpu_options()->set_allocator_type("BFC");

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

  string tf_model_name = fs::path(FLAGS_tf_model).remove_trailing_separator().stem().string();

  Expects(!FLAGS_run_id.empty());

  auto out_dir =
      mvshape::FileIO::FullOutPath((fs::path("tf_out") / tf_model_name / FLAGS_run_id / "checkpoints").string());

  if (!fs::exists(out_dir)) {
    fs::create_directories(out_dir);
    LOG(INFO) << "mkdir -p " << out_dir;
  }

  string out_file = (fs::path(out_dir) / name).string();

  filename_tensor.matrix<std::string>()(0, 0) = out_file;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
      {"saver/Const:0", filename_tensor},  // Filename
  };
  LOG(INFO) << "Saving " << out_file;

  // https://stackoverflow.com/a/37671613
  TF_CHECK_OK(session->Run(feed, {}, {"saver/control_dependency"}, nullptr));

  LOG(INFO) << "Done";

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

  LOG(INFO) << "Restored checkpoint.";
  LOG(INFO) << "=== Global step " << step_epoch[0] << ", epoch: " << step_epoch[1] << " in " << filename << " ===";

  return step_epoch[0];
};

int IncrementEpochCount(tf::Session *session) {
  TF_CHECK_OK(session->Run({}, {}, {"increment_epoch"}, nullptr));
  return ScalarOutput<int>(session, "epoch");
}

string FindAndPrepareOutputDirectory(tf::Session *session, const string &name) {
  vector<int> epoch_step = ScalarOutput<int>(session, vector<string>{"epoch", "global_step"});
  auto step_name = WithLeadingZeros(epoch_step[0], 4) + "_" + WithLeadingZeros(epoch_step[1], 9);
  string tf_model_name = fs::path(FLAGS_tf_model).remove_trailing_separator().stem().string();
  auto out_dir = mvshape::FileIO::FullOutPath((fs::path("tf_out") / fs::basename(tf_model_name) /
      FLAGS_run_id / "tensors" / step_name).string());
  string out_file = (fs::path(out_dir) / (name + ".bin")).string();

  // name could contain a slash so out_dir might not be the parent directory.
  FileIO::PrepareDir(fs::path(out_file).parent_path().string());
  return out_file;
}

string SaveTensor(tf::Session *session, const tf::Tensor &tensor, const string &name) {
  Expects(!name.empty());

  string out_file = FindAndPrepareOutputDirectory(session, name);


  LOG(INFO) << "Saving " << out_file;

  mvshape::FileIO::SerializeTensor<float>(
      out_file, tensor.flat<float>().data(), mvshape::tf_utils::TensorShapeAsVector(tensor.shape()));

  return out_file;
}

std::pair<int, int> ParseCheckpointFilename(const string &filename) {
  auto stem = fs::path(filename).stem();
  Expects(!stem.empty());

  vector<string> parts;
  boost::split(parts, stem.string(), boost::is_any_of("_"));
  Expects(parts.size() == 2);

  int epoch = std::stoi(parts[0]);
  int global_step = std::stoi(parts[1]);

  return {epoch, global_step};
}

string FindLastCheckpoint(const string &checkpoint_dir) {
  auto files = mvshape::FileIO::RegularFilesInDirectory(checkpoint_dir);
  string filename;
  int global_step = std::numeric_limits<int>::min();

  for (const auto &file : files) {
    if (boost::algorithm::ends_with(file, ".index")) {
      auto epoch_and_global_step = mvshape::tf_utils::ParseCheckpointFilename(file);
      if (epoch_and_global_step.second > global_step) {
        global_step = epoch_and_global_step.second;
        filename = file;
      }
    }
  }

  if (global_step < 0 || global_step == std::numeric_limits<int>::min()) {
    LOG(INFO) << "Checkpoint file not found.";
    filename = "";
  }

  if (boost::algorithm::ends_with(filename, ".index")) {
    filename = filename.substr(0, filename.find_last_of("."));
  }

  return filename;
}

string FindCheckpointAtEpoch(int epoch) {
  string tf_model_name = fs::path(FLAGS_tf_model).remove_trailing_separator().stem().string();
  auto checkpoint_dir = mvshape::FileIO::FullOutPath((fs::path("tf_out")
      / tf_model_name / FLAGS_run_id / "checkpoints").string());
  auto files = mvshape::FileIO::RegularFilesInDirectory(checkpoint_dir);

  Expects(std::is_sorted(std::begin(files), std::end(files)));

  string filename = "";

  for (const auto &file : files) {
    if (boost::algorithm::ends_with(file, ".index")) {
      auto epoch_and_global_step = mvshape::tf_utils::ParseCheckpointFilename(file);
      if (epoch_and_global_step.first == epoch) {
        filename = file;
        break;
      }
    }
  }

  if (filename.empty()) {
    LOG(ERROR) << "Checkpoint at epoch " << epoch << " not found in " << checkpoint_dir;
    throw std::runtime_error("Checkpoint file not found.");
  }

  if (boost::algorithm::ends_with(filename, ".index")) {
    filename = filename.substr(0, filename.find_last_of("."));
  }

  return filename;
}

string FindLastCheckpoint() {
  string tf_model_name = fs::path(FLAGS_tf_model).remove_trailing_separator().stem().string();
  auto checkpoing_dir = mvshape::FileIO::FullOutPath((fs::path("tf_out")
      / tf_model_name / FLAGS_run_id / "checkpoints").string());
  return FindLastCheckpoint(checkpoing_dir);
}

bool DidChange(int value) {
  static int prev = value;
  bool changed = value != prev;
  prev = value;
  return changed;
}

std::map<int, mv::Examples> SplitExamplesByTags(const mv::Examples &examples,
                                                const std::unordered_set<int> &tags) {
  Expects(examples.examples_size() > 0);

  std::map<int, mv::Examples> examples_by_tag;

  for (const auto &example :examples.examples()) {
    for (const auto &tag :example.tags()) {
      if (tags.find(tag) != tags.end()) { // Exists.
        examples_by_tag[tag].add_examples()->CopyFrom(example);
      }
    }
  }

  // TODO: extract common tags.

  std::stringstream stream;
  stream << "SplitExamplesByTags: ";

  for (const auto &tag: tags) {
    auto &e = examples_by_tag[tag];
    e.set_split_name(examples.split_name());
    e.set_dataset_name(examples.dataset_name());

    stream << mv::Tag_Name(static_cast<mv::Tag >(tag)) << "("
           << e.examples_size() << ") ";
  }
  LOG(INFO) << stream.str();
  return examples_by_tag;
};

}
}
