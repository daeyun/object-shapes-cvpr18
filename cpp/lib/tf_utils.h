//
// Created by daeyun on 7/12/17.
//

#pragma once

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/framework/tensor.h"

#include "common.h"

namespace mvshape {
namespace tf_utils {
namespace tf = tensorflow;

int LoadSavedModel(const std::unordered_set<std::string> &tags, tf::SavedModelBundle *model);

tf::TensorShape VectorAsTensorShape(const vector<tf::int64> &shape);

vector<int> TensorShapeAsVector(const tf::TensorShape &shape);

template<typename T>
void SetTensorData(const string &data, tf::Tensor *tensor) {
  int num_elements = static_cast<int>(data.size()) / sizeof(T);
  Expects(num_elements == tensor->shape().num_elements());
  Expects(tensor->AllocatedBytes() == data.size());
  std::memcpy(tensor->flat<T>().data(), data.data(), data.size());
}

template<tf::DataType DT = tf::DT_FLOAT, typename T = typename tf::EnumToDataType<DT>::Type>
tf::Tensor MakeTensor(const std::vector<long long> &shape, const std::string &data = "") {
  auto tensor = tf::Tensor(DT, VectorAsTensorShape(shape));
  if (!data.empty()) {
    SetTensorData<T>(data, &tensor);
  }
  return tensor;
}

template<tf::DataType DT = tf::DT_FLOAT, typename T, typename TT = typename tf::EnumToDataType<DT>::Type>
tf::Tensor MakeScalarTensor(T value) {
  auto tensor = tf::Tensor(DT, tf::TensorShape());;
  tensor.scalar<TT>()() = value;
  return tensor;
}

template<typename T>
vector<T> ScalarOutput(tf::Session *session, const vector<string> &names) {
  std::vector<tf::Tensor> out;
  TF_CHECK_OK(session->Run({}, names, {}, &out));
  vector<T> ret;
  ret.reserve(out.size());
  for (const auto &tensor : out) {
    Expects(0 == tensor.dims());
    ret.push_back(tensor.scalar<T>()());
  }
  return ret;
}

template<typename T>
vector<T> ScalarOutput(tf::Session *session,
                       const vector<string> &names,
                       const std::vector<std::pair<std::string, tensorflow::Tensor>> &feed) {
  std::vector<tf::Tensor> out;
  TF_CHECK_OK(session->Run(feed, names, {}, &out));
  vector<T> ret;
  ret.reserve(out.size());
  for (const auto &tensor : out) {
    Expects(0 == tensor.dims());
    ret.push_back(tensor.scalar<T>()());
  }
  return ret;
}

template<typename T>
T ScalarOutput(tf::Session *session, const string &name) {
  return ScalarOutput<T>(session, vector<string>{name})[0];
}

template<typename T>
T ScalarOutput(tf::Session *session,
               const string &name,
               const std::vector<std::pair<std::string, tensorflow::Tensor>> &feed) {
  return ScalarOutput<T>(session, vector<string>{name}, feed)[0];
}

string SaveCheckpoint(tf::Session *session);

int RestoreCheckpoint(tf::Session *session, const string &filename);

int IncrementEpochCount(tf::Session *session);

static bool DidChange(int value) {
  static int prev = value;
  bool changed = value!=prev;
  prev = value;
  return changed;
}


}
}
