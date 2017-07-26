//
// Created by daeyun on 6/30/17.
//

#include <algorithm>
#include <list>

#include <gtest/gtest.h>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/graph/default_device.h"

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

  auto session_options = tf::SessionOptions();
  session_options.config.mutable_device_count()->clear();
  session_options.config.mutable_device_count()->insert({"GPU", 0});
  session_options.config.mutable_gpu_options()->set_visible_device_list("");
  session_options.config.set_allow_soft_placement(true);

  ClientSession session(root, session_options);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]

  EXPECT_FLOAT_EQ(19, outputs[0].matrix<float>()(0));
  EXPECT_FLOAT_EQ(-3, outputs[0].matrix<float>()(1));
}
