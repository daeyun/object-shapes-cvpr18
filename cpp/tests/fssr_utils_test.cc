//
// Created by daeyun on 6/7/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>
#include <cpp/lib/batch_queue.h>

#include "file_io.h"
#include "fssr_utils.h"
#include "resources.h"

using namespace mvshape;

// TODO: (fssr_utils is no longer being developed. needs cleanup)

//TEST(FssrUtils, WriteMvei) {
//  auto filename = Resources::ResourcePath("tf_out/0040_000011400/NOVELCLASS/placeholder_target_depth/0.bin");
//  vector<int> shape;
//  vector<float> data;
//  FileIO::ReadTensorData(filename, &shape, &data);
//  data.resize(128 * 128);
//  shape = {shape[2], shape[3], shape[4]};
//
//  const auto out_filename = FileIO::FullOutPath("view_0.mvei");
//  LOG(INFO) << "Writing " << out_filename;
//
//  EXPECT_FALSE(FileIO::Exists(out_filename));
//
//  fssr::WriteMvei<float>(out_filename, shape, data);
//  EXPECT_TRUE(FileIO::Exists(out_filename));
//}
//
//TEST(FssrUtils, ReadTensors) {
//  auto tensor_dir = Resources::ResourcePath("tf_out/0040_000011400/NOVELCLASS");
//  mvshape::concurrency::BatchQueue<fssr::FloatImages> queue(4);
//  auto thread = fssr::ReadSavedTensors(tensor_dir, {
//      "placeholder_target_depth",
//      "out_depth",
//      "out_silhouette",
//  }, &queue);
//
//  fssr::FloatImages tensors;
//  queue.Dequeue(&tensors, 1);
//
//  EXPECT_EQ(3, tensors.images.size());
//
//  LOG(INFO) << tensors.index;
//  for (const auto &item : tensors.images) {
//    LOG(INFO) << item.first;
//  }
//
//  EXPECT_TRUE(std::isnan(tensors.images["placeholder_target_depth"][0]->at(0)));
//
//  queue.Close();
//  thread.join();
//}
//
//TEST(FssrUtils, Depth2Mesh) {
//  auto tensor_dir = Resources::ResourcePath("tf_out/0040_000011400/NOVELCLASS");
//  auto out_dir = FileIO::FullOutPath("recon/0040_000011400/NOVELCLASS");
//  fssr::Depth2Mesh(tensor_dir, out_dir);
//}





