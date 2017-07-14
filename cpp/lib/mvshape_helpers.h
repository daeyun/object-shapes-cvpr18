//
// Created by daeyun on 7/13/17.
//

#pragma once

#include "tensorflow/core/framework/tensor.h"
#include <boost/filesystem.hpp>

#include "cpp/lib/tf_utils.h"
#include "proto/dataset.pb.h"

namespace mvshape {
namespace tf = tensorflow;
namespace mv = mvshape_dataset;
namespace fs = boost::filesystem;

struct Shrec12Placeholders {
  static Shrec12Placeholders Build(int batch_size) {
    Shrec12Placeholders ret;
    ret.is_training = tf_utils::MakeScalarTensor<tf::DT_BOOL>(false);
    ret.target_depth_offset = tf_utils::MakeScalarTensor<tf::DT_FLOAT>(-5.5);
    ret.in_depth = tf::Tensor(tf::DT_FLOAT, {batch_size, 128, 128, 1});
    ret.target_depth = tf::Tensor(tf::DT_FLOAT, {batch_size, 6, 128, 128, 1});
    return ret;
  }

  tf::Tensor target_depth_offset;
  tf::Tensor is_training;
  tf::Tensor in_depth;
  tf::Tensor target_depth;
};

namespace evaluation {

std::map<string, float> Shrec12(tf::Session *session, const mv::Examples &eval_examples);

}
}
