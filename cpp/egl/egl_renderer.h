//
// Created by daeyun on 6/14/17.
//
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>

namespace mvshape {
namespace Rendering {

using CamMat = Eigen::Matrix<float, 4, 4, Eigen::ColMajor>;
using std::vector;
using std::array;
using std::unique_ptr;

std::string ToHexString(int num);

void CheckGlError();

void CheckEglError();

unsigned int LoadShader(const char *shader_source, int type);

struct RendererConfig {
  int width;
  int height;
  std::string vert_shader_src;
  std::string frag_shader_src;
  std::string geom_shader_src;
};

struct FrameConfig {
  CamMat view_matrix;
  CamMat proj_matrix;
  float scale = 1.0;
};

class Renderer {
 public:
  explicit Renderer(const RendererConfig &config) : config_(config) {}

  virtual void Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
                      vector<vector<std::unique_ptr<float[]>>> *out_frames) = 0;

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

 protected:
  void CreateEglContext();
};

class ShapeRendererBase : public Renderer {
 public:
  explicit ShapeRendererBase(const RendererConfig &config);

  // `out_frames` is a 2D vector of [camera_index][shader_output_channel]
  void Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
              vector<vector<std::unique_ptr<float[]>>> *out_frames) override;

  void Cleanup() override;

 protected:
  void InitializeFramebuffers() override;
  void InitializeShaderProgram() override;

  unsigned int depth_test_buffer_;
  unsigned int depth_texture_buffer_;
  unsigned int normal_texture_buffer_;
};
}
}
