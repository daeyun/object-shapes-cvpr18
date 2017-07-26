//
// Created by daeyun on 6/14/17.
//

#include "egl_renderer.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <GL/gl.h>

#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <memory>


namespace mvshape {
namespace Rendering {

void ShapeRendererBase::InitializeFramebuffers() {
  // Framebuffer for off-screen rendering.
  glGenFramebuffers(1, &frame_buffer_);

  // Buffer used for depth testing.
  glGenRenderbuffers(1, &depth_test_buffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_test_buffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, config_.width, config_.height);

  // Texture buffer 0.
  glGenTextures(1, &depth_texture_buffer_);
  glBindTexture(GL_TEXTURE_2D, depth_texture_buffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, config_.width, config_.height, 0, GL_RED, GL_FLOAT, nullptr);

  // Texture buffer 1.
  glGenTextures(1, &normal_texture_buffer_);
  glBindTexture(GL_TEXTURE_2D, normal_texture_buffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, config_.width, config_.height, 0, GL_RGBA, GL_FLOAT, nullptr);

  // Attach buffers.
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_test_buffer_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depth_texture_buffer_, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normal_texture_buffer_, 0);

  // Set the list of draw buffers.
  GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, DrawBuffers);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "Framebuffer status: " << status << std::endl;
  }
}

void ShapeRendererBase::InitializeShaderProgram() {
  auto vertex_src = config_.vert_shader_src;
  GLuint vertexShader = LoadShader(vertex_src.c_str(), GL_VERTEX_SHADER);

  auto frag_src = config_.frag_shader_src;
  GLuint fragmentShader = LoadShader(frag_src.c_str(), GL_FRAGMENT_SHADER);

  auto geom_src = config_.geom_shader_src;
  GLuint geomShader = LoadShader(geom_src.c_str(), GL_GEOMETRY_SHADER);

  shader_program_ = glCreateProgram();
  glAttachShader(shader_program_, vertexShader);
  glAttachShader(shader_program_, fragmentShader);
  glAttachShader(shader_program_, geomShader);

  glLinkProgram(shader_program_);
}

void ShapeRendererBase::Render(const vector<mvshape::Rendering::FrameConfig> &frame_configs,
                           vector<vector<std::unique_ptr<float[]>>> *out_frames) {
  if (eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_) == EGL_FALSE) {
    std::cerr << "eglMakeCurrent failed." << std::endl;
    CheckEglError();
  }
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
  glUseProgram(shader_program_);

  // Initialize viewport.
  glViewport(0, 0, config_.width, config_.height);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_TRUE);

  // Bind data buffer.
  glEnableVertexAttribArray(kVertexShaderIndex);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glVertexAttribPointer(
      kVertexShaderIndex, // attribute 0. must match the layout in the shader.
      3,                  // size
      GL_FLOAT,           // type
      GL_FALSE,           // normalized?
      0,                  // stride
      (void *) 0          // array buffer offset
  );

  for (const auto &frame_config : frame_configs) {
    float scale = frame_config.scale;

    // Reset.
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Eigen::Matrix<float, 4, 4, Eigen::ColMajor> MV = frame_config.view_matrix;
    Eigen::Matrix<float, 4, 4, Eigen::ColMajor> P = frame_config.proj_matrix;

    GLint modelview_id = glGetUniformLocation(shader_program_, "modelview");
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, MV.data());

    GLint projection_id = glGetUniformLocation(shader_program_, "projection");
    glUniformMatrix4fv(projection_id, 1, GL_FALSE, P.data());

    GLint scale_id = glGetUniformLocation(shader_program_, "scale");
    glUniform1f(scale_id, scale);

    // Draw the triangles.
    glDrawArrays(GL_TRIANGLES, 0, num_triangles_ * 3); // Starting from vertex 0; n*3 vertices total
    glFinish();

    const auto num_pixels = config_.width*config_.height;
    // Read.
    auto depth_image_buffer = std::unique_ptr<float[]>(new float[num_pixels]());
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, config_.width, config_.height, GL_RED, GL_FLOAT, depth_image_buffer.get());
    CheckGlError();

    // TODO
//    cv::flip(*depth_image, *depth_image, 0);

    auto normal_image_buffer = std::unique_ptr<float[]>(new float[num_pixels*3]());
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, config_.width, config_.height, GL_RGB, GL_FLOAT, normal_image_buffer.get());
    CheckGlError();

    // TODO
//    cv::flip(*normal_image, *normal_image, 0);

    vector<std::unique_ptr<float[]>> frame;
    frame.emplace_back(std::move(depth_image_buffer));
    frame.emplace_back(std::move(normal_image_buffer));

    out_frames->emplace_back(std::move(frame));
  }

  glDisableVertexAttribArray(kVertexShaderIndex);
}

void ShapeRendererBase::Cleanup() {
  glDeleteTextures(1, &depth_texture_buffer_);
  glDeleteTextures(1, &normal_texture_buffer_);
  glDeleteRenderbuffers(1, &depth_test_buffer_);

  glDeleteBuffers(1, &vertex_buffer_);
  glDeleteFramebuffers(1, &frame_buffer_);
  glDeleteProgram(shader_program_);

  eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroySurface(egl_display_, egl_surface_);
  eglDestroyContext(egl_display_, egl_context_);

  // Default display.
  eglTerminate(egl_display_);
}

ShapeRendererBase::ShapeRendererBase(const RendererConfig &config) : Renderer(config) {
  Initialize();
}

std::string ToHexString(int num) {
  std::stringstream stream;
  stream << std::hex << num;
  return "0x" + std::string(stream.str());
}

void CheckGlError() {
  auto error = glGetError();
  if (error != GL_NO_ERROR) {
    throw std::runtime_error("GL error code: " + ToHexString(error));
  }
}

void CheckEglError() {
  auto error = eglGetError();
  if (error != EGL_SUCCESS) {
    throw std::runtime_error("EGL error code: " + ToHexString(error));
  }
}

unsigned int LoadShader(const char *shader_source, int type) {
  GLuint shader = glCreateShader(static_cast<GLenum>(type));
  if (shader == 0) {
    throw std::runtime_error("Could not create shader. type: " + std::to_string(type));
  }

  glShaderSource(shader, 1, &shader_source, nullptr);
  CheckGlError();

  glCompileShader(shader);

  GLint status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

  if (status == GL_FALSE) {
    std::cerr << "Shader compilation error. Source:" << std::endl << shader_source << std::endl;

    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    std::string errorLog(maxLength, '\0');
    glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

    glDeleteShader(shader); // Don't leak the shader.
  }

  return shader;
}

void Renderer::SetTriangleVertices(const std::vector<std::array<std::array<float, 3>, 3>> &triangles) {
  const GLfloat *data_ptr = &triangles[0][0][0];
  num_triangles_ = static_cast<int>(triangles.size());

  if (vertex_buffer_ == UINT32_MAX) {
    // Data buffer for triangle vertices.
    glGenBuffers(1, &vertex_buffer_);
  }
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  // Makes a copy of `triangles`.
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * num_triangles_ * 3 * 3, data_ptr, GL_STATIC_DRAW);
  has_triangles_ = true;
}

void Renderer::CreateEglContext() {
  // 1. Initialize EGL
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  if (display == EGL_NO_DISPLAY) {
    throw std::runtime_error("eglGetDisplay failed.");
  }

  EGLint major, minor;
  if (!eglInitialize(display, &major, &minor)) {
    std::cerr << "eglInitialize failed." << std::endl;
    CheckEglError();
  }

  // 2. Select an appropriate configuration
  const EGLint configAttribs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_BLUE_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_RED_SIZE, 8,
      EGL_DEPTH_SIZE, 24,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_NONE
  };

  EGLint num_configs;
  EGLConfig egl_config;

  if (!eglChooseConfig(display, configAttribs, &egl_config, 1, &num_configs)) {
    std::cerr << "eglChooseConfig failed." << std::endl;
    CheckEglError();
  }

  const EGLint buffer_attribs[] = {
      EGL_WIDTH, this->config_.width,
      EGL_HEIGHT, this->config_.width,
      EGL_NONE,
  };

  // 3. Create a surface
  EGLSurface surface = eglCreatePbufferSurface(display, egl_config, buffer_attribs);
  if (surface == EGL_NO_SURFACE) {
    std::cerr << "Could not create EGL surface." << std::endl;
    CheckEglError();
  }

  // 4. Bind the API
  if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
    std::cerr << "eglBindAPI failed." << std::endl;
    CheckEglError();
  }

  // 5. Create a context and make it current
  EGLContext context = eglCreateContext(display, egl_config, EGL_NO_CONTEXT, nullptr);
  if (context == EGL_NO_CONTEXT) {
    std::cerr << "eglCreateContext failed." << std::endl;
    CheckEglError();
  }

  if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
    std::cerr << "eglMakeCurrent failed." << std::endl;
    CheckEglError();
  }

  egl_context_ = context;
  egl_surface_ = surface;
  egl_display_ = display;
}











}
}








