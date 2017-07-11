//
// Created by daeyun on 6/14/17.
//
#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "camera.h"

namespace mvshape {
namespace Rendering {

constexpr char kVertShaderPath[] = "shaders/object.vert";
constexpr char kGeomShaderPath[] = "shaders/object.geom";
constexpr char kFragShaderPath[] = "shaders/object.frag";

static const int pbufferWidth = 512;
static const int pbufferHeight = 512;

std::string ToHexString(int num);

void CheckGlError();

void CheckEglError();

unsigned int LoadShader(const char *shader_source, int type);

struct RendererConfig {
  int width;
  int height;
};

struct FrameConfig {
  mvshape::Camera *camera;
  float scale = 1.0;
};

class Renderer {
 public:
  explicit Renderer(const RendererConfig &config) : config_(config) {}

  virtual void Render(const vector<mvshape::Camera *> &cameras, vector<vector<unique_ptr<cv::Mat>>> *out_frames) = 0;

  virtual void Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
                      vector<vector<unique_ptr<cv::Mat>>> *out_frames) = 0;

  void SetTriangleVertices(const std::vector<std::array<std::array<float, 3>, 3>> &triangles);

  bool has_triangles() const {
    return has_triangles_;
  }

  static constexpr unsigned int kVertexShaderIndex = 0;

  virtual void Cleanup() = 0;

 protected:
  // Sets `frame_buffer_`.
  virtual void InitializeFramebuffers() = 0;
  // Sets `shader_program_`.
  virtual void InitializeShaderProgram() = 0;

  void Initialize() {
    CreateEglContext();
    InitializeFramebuffers();
    InitializeShaderProgram();
  }

  bool has_triangles_ = false;

  const RendererConfig config_;
  void *egl_context_;
  void *egl_surface_;
  void *egl_display_;
  int num_triangles_ = 0;
  unsigned int frame_buffer_;
  unsigned int vertex_buffer_ = UINT32_MAX;
  unsigned int shader_program_;

 private:
  void CreateEglContext();
};

class ShapeRenderer : public Renderer {
 public:
  ShapeRenderer(const RendererConfig &config);

  // `out_frames` is a 2D vector of [camera_index][shader_output_channel]
  virtual void Render(const vector<mvshape::Camera *> &cameras,
                      vector<vector<unique_ptr<cv::Mat>>> *out_frames) override;

  virtual void Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
                      vector<vector<unique_ptr<cv::Mat>>> *out_frames) override;

  void Cleanup();

 private:
  virtual void InitializeFramebuffers() override;
  virtual void InitializeShaderProgram() override;

  unsigned int depth_test_buffer_;
  unsigned int depth_texture_buffer_;
  unsigned int normal_texture_buffer_;
};
}
}

