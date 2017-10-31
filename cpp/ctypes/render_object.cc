#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define GSL_THROW_ON_CONTRACT_VIOLATION
#include <gsl/gsl_assert>
#include <glog/logging.h>

#include "cpp/lib/egl_rendering.h"

DECLARE_string(resources_dir);

// Interface for Python ctypes.
extern "C" {
void depth_and_normal_map(const float *vertices, const size_t len_vertices,
                          const float *eye, const float *center, const float *up,
                          const int w, const int h,
                          const char *path_to_resources,
                          float *out_depth, float *out_normal);
}

void depth_and_normal_map(const float *vertices, const size_t len_vertices,
                          const float *eye, const float *center, const float *up,
                          const int w, const int h,
                          const char *path_to_resources,
                          float *out_depth, float *out_normal) {
  Expects(len_vertices % 9 == 0);

  if (path_to_resources != nullptr) {
    FLAGS_resources_dir = std::string(path_to_resources);
  }

  using namespace mvshape::Rendering;
  vector<array<array<float, 3>, 3>> triangles;

  const float *v = vertices;
  for (int i = 0; i < len_vertices; i += 9) {
    triangles.push_back(array<array<float, 3>, 3> {
        array<float, 3> {*(v + 0), *(v + 1), *(v + 2)},
        array<float, 3> {*(v + 3), *(v + 4), *(v + 5)},
        array<float, 3> {*(v + 6), *(v + 7), *(v + 8)}
    });
    v += 9;
  }

  RendererConfig config{
      .width = w,
      .height = h,
  };

  auto renderer = ShapeRenderer(config);

  renderer.SetTriangleVertices(triangles);

  mvshape::FrustumParams frustum;
  frustum.near = 0.1;
  frustum.far = 100;

  auto camera = mvshape::OrthographicCamera(Vec3{*eye, *(eye + 1), *(eye + 2)},
                                            Vec3{*center, *(center + 1), *(center + 2)},
                                            Vec3{*up, *(up + 1), *(up + 2)},
                                            frustum);

  vector<vector<unique_ptr<cv::Mat>>> out_frames;
  renderer.Render({&camera}, &out_frames);
  Expects(out_frames.size() == 1);

  uchar *data = out_frames[0][0]->data;
  auto size = out_frames[0][0]->total() * out_frames[0][0]->elemSize();
  Expects(size == w * h * sizeof(float));

  memcpy(out_depth, data, size);
}
