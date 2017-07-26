//
// Created by daeyun on 6/14/17.
//
#pragma once

#include <opencv2/opencv.hpp>

#include "common.h"
#include "camera.h"
#include "cpp/egl/egl_renderer.h"

namespace mvshape {
namespace Rendering {

constexpr char kVertShaderPath[] = "shaders/object.vert";
constexpr char kGeomShaderPath[] = "shaders/object.geom";
constexpr char kFragShaderPath[] = "shaders/object.frag";

class ShapeRenderer : public ShapeRendererBase {
 public:
  ShapeRenderer(const RendererConfig &config) : ShapeRendererBase(config) {}

  // `out_frames` is a 2D vector of [camera_index][shader_output_channel]
  void Render(const vector<mvshape::Camera *> &cameras,
              vector<vector<unique_ptr<cv::Mat>>> *out_frames);

  void Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
              vector<vector<unique_ptr<cv::Mat>>> *out_frames);
};
}
}

