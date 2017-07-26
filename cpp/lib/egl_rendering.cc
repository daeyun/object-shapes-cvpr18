//
// Created by daeyun on 6/14/17.
//

#include "egl_rendering.h"

#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>

#include "common.h"

#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "resources.h"

namespace mvshape {
namespace Rendering {

void ShapeRenderer::Render(const vector<mvshape::Camera *> &cameras, vector<vector<unique_ptr<cv::Mat>>> *out_frames) {
  vector<mvshape::Rendering::FrameConfig> frame_configs;
  for (const auto &camera : cameras) {
    frame_configs.push_back(mvshape::Rendering::FrameConfig {
        .view_matrix = camera->view_mat().cast<float>(),
        .proj_matrix = camera->projection_mat().cast<float>(),
    });
  }

  Render(frame_configs, out_frames);

}

void ShapeRenderer::Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
                           vector<vector<unique_ptr<cv::Mat>>> *out_frames) {
  vector<vector<std::unique_ptr<float[]>>> out_frames_raw;
  ShapeRendererBase::Render(frame_configs, &out_frames_raw);

  // TODO
}

}
}
