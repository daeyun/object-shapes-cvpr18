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

DEFINE_int32(batch_size, 60, "Number of batches per training step.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  Expects(!FLAGS_run_id.empty());

  tf::SavedModelBundle model;
  tf::graph::SetDefaultDevice("/gpu:0", model.meta_graph_def.mutable_graph_def());

  int global_step = mvshape::tf_utils::LoadSavedModel({"mv"}, &model);
  tf::graph::SetDefaultDevice("/gpu:0", model.meta_graph_def.mutable_graph_def());

  Expects(0 == global_step);



  //////////////////////////////////////////
  mvshape_dataset::Examples train_examples;
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/train.bin"), &train_examples);
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/validation.bin"),
                              &train_examples);

  mvshape_dataset::Examples test_examples;
  mvshape::Data::LoadExamples(mvshape::FileIO::FullOutPath("splits/shrec12_examples_vpo/test.bin"), &test_examples);

  std::map<int, mv::Examples> test_examples_by_tag;

  for (const auto &example :test_examples.examples()) {
    for (const auto &tag :example.tags()) {
      switch (tag) {
        case mv::NOVELVIEW:
        case mv::NOVELMODEL:
        case mv::NOVELCLASS:
          test_examples_by_tag[tag].add_examples()->CopyFrom(example);
          break;
        default:
          break;
      }
      break;
    }
  }
  test_examples_by_tag[mv::NOVELVIEW].set_split_name(mv::TEST);
  test_examples_by_tag[mv::NOVELMODEL].set_split_name(mv::TEST);
  test_examples_by_tag[mv::NOVELCLASS].set_split_name(mv::TEST);
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
  }, batch_size, true, true);

  loader.StartWorkers();

  mvshape::Timer timer("train");

  while (true) {
    auto batch = loader.Next();
    // TODO
    if (batch == nullptr) {
      break;
    }

    timer.Toc();

    is_training.scalar<bool>()() = true;
    tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kSingleDepthFieldNumber), &in_depth);
    tf_utils::SetTensorData<float>(batch->file_fields.at(mv::Example::kMultiviewDepthFieldNumber), &target_depth);

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
        {"placeholder/is_training", is_training},
        {"placeholder/target_depth_offset", target_depth_offset},
        {"placeholder/in_depth", in_depth},
        {"placeholder/target_depth", target_depth},
    };

    vector<tf::Tensor> out;
    TF_CHECK_OK(model.session->Run(feed, {"iou"}, {"train_op"}, &out));

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
      LOG(INFO) << "Starting evaluation at " << loader.epoch();

      for (int tag: {mv::NOVELVIEW, mv::NOVELMODEL, mv::NOVELCLASS}) {
        mvshape::Data::BatchLoader eval_loader(&test_examples_by_tag[tag], {
            mv::Example::kSingleDepthFieldNumber,
            mv::Example::kMultiviewDepthFieldNumber,
        }, batch_size, false, false);

        eval_loader.StartWorkers();

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

          auto result = tf_utils::ScalarOutput<float>(model.session.get(), names, feed);
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

      tf_utils::SaveCheckpoint(model.session.get());

      tf_utils::IncrementEpochCount(model.session.get());
    }  // end of if(DidChange(epoch))

  }

#if 0


  int save_count = 0;

  while (true) {
    auto batch = loader.Next();
    // TODO
    if (batch == nullptr) {
      break;
    }

    tf::Tensor training(tf::DT_BOOL, tf::TensorShape());
    tf::Tensor target_depth_offset(tf::DT_FLOAT, tf::TensorShape());
    tf::Tensor in_depth(tf::DT_FLOAT, {batch->size, 128, 128, 1});
    tf::Tensor target_depth(tf::DT_FLOAT, {batch->size, 6, 128, 128, 1});

    // shrec12 specific.
    target_depth_offset.scalar<float>()() = -5.5;

    training.scalar<bool>()() = true;
    mvshape::SetTensorData<float>(batch->file_fields.at(mv::Example::kSingleDepthFieldNumber), &in_depth);
    mvshape::SetTensorData<float>(batch->file_fields.at(mv::Example::kMultiviewDepthFieldNumber), &target_depth);

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed = {
        {"placeholder/is_training", training},
        {"placeholder/target_depth_offset", target_depth_offset},
        {"placeholder/in_depth", in_depth},
        {"placeholder/target_depth", target_depth},
    };

    status = model.session->Run(feed, {"loss", "loss_d", "loss_s", "iou"}, {"train_op"}, &outputs);
    if (!status.ok()) {
      LOG(ERROR) << status.ToString();
      return 1;
    }
#ifndef NDEBUG
    for (const auto &output : outputs) {
      LOG(INFO) << output.DebugString();
    }
#endif

    timer.Toc();

    LOG(INFO) << "# " << loader.epoch() << ": " << loader.num_examples_returned_in_current_epoch() << " / "
              << loader.size()
              << ", loss: " << outputs[0].scalar<float>()
              << ", loss_d: " << outputs[1].scalar<float>()
              << ", loss_s: " << outputs[2].scalar<float>()
              << ", iou: " << outputs[3].scalar<float>();
    int global_step = get_global_step();
    LOG(INFO) << "global step: " << global_step;

    LOG(INFO) << "Elapsed: " << std::fixed << std::setprecision(2) << timer.Elapsed() << " seconds.";

    if (save_count <= loader.num_examples_returned() / 10000) {

      std::vector<string> fetch = {
          "in_depth_zeroed",
          "in_silhouette",
          "target_depth_zeroed",
          "target_silhouette",
          "out_depth",
          "out_silhouette",
      };

      training.scalar<bool>()() = false;
      status = model.session->Run(feed, fetch, {}, &outputs);
      if (!status.ok()) {
        LOG(ERROR) << status.ToString();
        return 1;
      }

      for (int i = 0; i < fetch.size(); ++i) {
        auto output = outputs[i];
        auto name = fetch[i];
        save_tensor(output, name);
      }
      // TODO
      save_checkpoint();

      save_count++;
    }
  }

#if 0
  mvshape::Data::BatchLoader eval_loader(&test_examples, {
      mv::Example::kSingleDepthFieldNumber,
      mv::Example::kMultiviewDepthFieldNumber,
  }, batch_size, false, false);

  eval_loader.StartWorkers();

  while(true){

  training.scalar<bool>()() = false;
  status = model.session->Run(feed, {"loss", "abc"}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status.ToString();
    return 1;
  }
  for (const auto &output : outputs) {
    LOG(INFO) << output.DebugString();
  }
  }

  eval_loader.StopWorkers();
#endif


#endif

  loader.StopWorkers();
  model.session->Close();

}
