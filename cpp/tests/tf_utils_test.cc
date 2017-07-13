//
// Created by daeyun on 6/30/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <algorithm>
#include <list>

#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/graph/default_device.h"

#include "tf_utils.h"
#include "flags.h"
#include "resources.h"
#include "file_io.h"
#include "random_utils.h"

namespace tf = tensorflow;

// Hello-world example from https://www.tensorflow.org/api_guides/cc/guide
TEST(Tensorflow, Basics) {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]

  EXPECT_FLOAT_EQ(19, outputs[0].matrix<float>()(0));
  EXPECT_FLOAT_EQ(-3, outputs[0].matrix<float>()(1));
}

TEST(Tensorflow, SimpleTraining) {
  // Setup.
  tf::SavedModelBundle model;

  FLAGS_tf_model = mvshape::Resources::ResourcePath("tf_models/small_trainable_model");
  int global_step = mvshape::tf_utils::LoadSavedModel({"small_trainable_model"}, &model);
  EXPECT_EQ(0, global_step);

  int batch_size = 3;

  vector<float> values;
  int size = 64 * 64 * batch_size;
  for (int i = 0; i < size; ++i) {
    float value = i / static_cast<float>(size);
    values.push_back(value);
  }

  string b = mvshape::FileIO::ToBytes<float>(values);
  EXPECT_LT(1000, b.size());

  auto input_tensor = mvshape::tf_utils::MakeTensor<tf::DT_FLOAT>({batch_size, 64, 64, 1});
  mvshape::tf_utils::SetTensorData<float>(b, &input_tensor);

  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(i / static_cast<float>(size), input_tensor.flat<float>()(i));
  }

  auto is_training = mvshape::tf_utils::MakeScalarTensor<tf::DT_BOOL>(true);
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
      {"placeholder/is_training", is_training},
      {"placeholder/images", input_tensor},
  };

  // Training losses should be decreasing.

  std::list<float> losses;

  for (int j = 0; j < 20; ++j) {
    vector<tf::Tensor> outputs;
    TF_CHECK_OK(model.session->Run(feed, {"loss"}, {"train_op"}, &outputs));
    losses.push_front(outputs[0].scalar<float>()());
  }

  EXPECT_TRUE(std::is_sorted(losses.begin(), losses.end()));

  // Making sure is_training is passed correctly.
  bool is_training_value;

  is_training.scalar<bool>()() = false;
  is_training_value = mvshape::tf_utils::ScalarOutput<bool>(model.session.get(), "identity/is_training", feed);
  EXPECT_FALSE(is_training_value);

  is_training.scalar<bool>()() = true;
  is_training_value = mvshape::tf_utils::ScalarOutput<bool>(model.session.get(), "identity/is_training", feed);
  EXPECT_TRUE(is_training_value);

  model.session->Close();
}

TEST(Tensorflow, Integration) {
  // Setup.

  tf::SavedModelBundle model;

  FLAGS_tf_model = mvshape::Resources::ResourcePath("tf_models/small_trainable_model");
  int global_step = mvshape::tf_utils::LoadSavedModel({"small_trainable_model"}, &model);
  EXPECT_EQ(0, global_step);

  int batch_size = 3;

  vector<float> values1;
  int size = 64 * 64 * batch_size;
  for (int i = 0; i < size; ++i) {
    float value = i / static_cast<float>(size);
    values1.push_back(value);
  }

  vector<float> values2;
  for (int i = size; i > 0; --i) {
    float value = i / static_cast<float>(size);
    values2.push_back(value);
  }

  string b1 = mvshape::FileIO::ToBytes<float>(values1);
  string b2 = mvshape::FileIO::ToBytes<float>(values2);
  EXPECT_EQ(b1.size(), b2.size());

  auto input_tensor = mvshape::tf_utils::MakeTensor<tf::DT_FLOAT>({batch_size, 64, 64, 1});

  auto is_training = mvshape::tf_utils::MakeScalarTensor<tf::DT_BOOL>(true);
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
      {"placeholder/is_training", is_training},
      {"placeholder/images", input_tensor},
  };

  is_training.scalar<bool>()() = false;

  // Not training. Output depends on the input. Repeatable.

  int global_step_1 = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "global_step");

  mvshape::tf_utils::SetTensorData<float>(b1, &input_tensor);
  float loss1 = mvshape::tf_utils::ScalarOutput<float>(model.session.get(), "loss", feed);

  mvshape::tf_utils::SetTensorData<float>(b2, &input_tensor);
  float loss2 = mvshape::tf_utils::ScalarOutput<float>(model.session.get(), "loss", feed);

  EXPECT_LT(1e-6, std::abs(loss1 - loss2));

  mvshape::tf_utils::SetTensorData<float>(b1, &input_tensor);
  float loss1_2 = mvshape::tf_utils::ScalarOutput<float>(model.session.get(), "loss", feed);

  mvshape::tf_utils::SetTensorData<float>(b2, &input_tensor);
  float loss2_2 = mvshape::tf_utils::ScalarOutput<float>(model.session.get(), "loss", feed);

  EXPECT_FLOAT_EQ(loss1, loss1_2);
  EXPECT_FLOAT_EQ(loss2, loss2_2);

  int global_step_2 = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "global_step");

  EXPECT_EQ(global_step_1, global_step_2);


  // When train_op runs with is_training=false, EMA won't be updated, but it should still train.

  vector<tf::Tensor> outputs;
  is_training.scalar<bool>()() = false;

  global_step = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "global_step");
  EXPECT_EQ(global_step, 0);

  mvshape::tf_utils::SetTensorData<float>(b1, &input_tensor);
  TF_CHECK_OK(model.session->Run(feed, {"loss"}, {"train_op"}, &outputs));
  loss1 = outputs[0].scalar<float>()();

  TF_CHECK_OK(model.session->Run(feed, {"loss"}, {"train_op"}, &outputs));
  loss1_2 = outputs[0].scalar<float>()();

  EXPECT_LT(1e-6, std::abs(loss1 - loss1_2));

  global_step = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "global_step");
  EXPECT_EQ(global_step, 2);


  // Save

  is_training.scalar<bool>()() = true;
  auto filename_1 = mvshape::tf_utils::SaveCheckpoint(model.session.get());
  LOG(INFO) << filename_1;

  TF_CHECK_OK(model.session->Run(feed, {"loss"}, {"train_op"}, &outputs));
  global_step = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "global_step");
  EXPECT_EQ(global_step, 3);

  auto filename_2 = mvshape::tf_utils::SaveCheckpoint(model.session.get());
  LOG(INFO) << filename_2;

  EXPECT_EQ(2, mvshape::tf_utils::RestoreCheckpoint(model.session.get(), filename_1));
  EXPECT_EQ(3, mvshape::tf_utils::RestoreCheckpoint(model.session.get(), filename_2));
  EXPECT_EQ(2, mvshape::tf_utils::RestoreCheckpoint(model.session.get(), filename_1));
}

TEST(Tensorflow, UpdateEpochCount) {
  tf::SavedModelBundle model;

  FLAGS_tf_model = mvshape::Resources::ResourcePath("tf_models/small_trainable_model");
  int global_step = mvshape::tf_utils::LoadSavedModel({"small_trainable_model"}, &model);
  EXPECT_EQ(0, global_step);

  int epoch = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "epoch");
  EXPECT_EQ(0, epoch);

  epoch = mvshape::tf_utils::IncrementEpochCount(model.session.get());
  EXPECT_EQ(1, epoch);

  epoch = mvshape::tf_utils::IncrementEpochCount(model.session.get());
  EXPECT_EQ(2, epoch);

  epoch = mvshape::tf_utils::ScalarOutput<int>(model.session.get(), "epoch");
  EXPECT_EQ(2, epoch);

  model.session->Close();
}

TEST(CheckpointIO, ParseFilename) {
  auto epoch_step = mvshape::tf_utils::ParseCheckpointFilename("/111/222/0000_00001044.index");
  EXPECT_EQ(0, epoch_step.first);
  EXPECT_EQ(1044, epoch_step.second);

  epoch_step = mvshape::tf_utils::ParseCheckpointFilename("/111/222/0003_00001044.index");
  EXPECT_EQ(3, epoch_step.first);
  EXPECT_EQ(1044, epoch_step.second);

  epoch_step = mvshape::tf_utils::ParseCheckpointFilename("/111/222/0003_00001044.data-0000-of-00001");
  EXPECT_EQ(3, epoch_step.first);
  EXPECT_EQ(1044, epoch_step.second);

  epoch_step = mvshape::tf_utils::ParseCheckpointFilename("/111/222/0003_00001044");
  EXPECT_EQ(3, epoch_step.first);
  EXPECT_EQ(1044, epoch_step.second);
}
