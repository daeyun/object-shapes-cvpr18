//
// Created by daeyun on 4/11/17.
//

#pragma once

#include "common.h"

namespace mvshape {
struct FrustumParams {
  // Same as gluPerspective.
  static FrustumParams MakePerspective(double fov_y, double aspect, double z_near, double z_far);

  double left = -1;
  float right = 1;
  double bottom = -1;
  double top = 1;
  double near = 1;
  double far = 10;
};

class Camera {
 public:
  Camera(const Vec3 &camera_position,
         const Vec3 &lookat_position,
         const Vec3 &up,
         const FrustumParams &frustum);

  void WorldToCam(const Vec3 &xyz, Vec3 *out) const;

  void CamToWorld(const Vec3 &xyz, Vec3 *out) const;

  void CamToWorldNormal(const Vec3 &xyz, Vec3 *out) const;

  void WorldToCamNormal(const Vec3 &xyz, Vec3 *out) const;

  void FrustumToCam(const Vec3 &xyz, Vec3 *out) const;

  void CamToFrustum(const Vec3 &xyz, Vec3 *out) const;

  const Mat44 &view_mat() const {
    return view_mat_;
  }

  const Mat44 &view_mat_inv() const {
    return view_mat_inv_;
  }

  const FrustumParams &frustum() const {
    return frustum_;
  }

  const Vec3 &position() const {
    return position_;
  }

  const Vec3 &lookat_position() const {
    return lookat_position_;
  }

  const Vec3 &up() const {
    return up_;
  }

  const Vec3 &viewing_direction() const {
    return viewing_direction_;
  }

  virtual const Mat44 &projection_mat() const = 0;
  virtual const Mat44 &projection_mat_inv() const = 0;

 private:
  Vec3 position_;
  Vec3 lookat_position_;
  Vec3 up_;
  float modelview_scale_ = 1.0;
  Vec3 viewing_direction_;
  Mat44 view_mat_;
  Mat44 view_mat_inv_;
  FrustumParams frustum_;
};

class OrthographicCamera : public Camera {
 public:
  OrthographicCamera(const Vec3 &camera_position,
                     const Vec3 &lookat_position,
                     const Vec3 &up,
                     const FrustumParams &frustum_params);

  const Mat44 &projection_mat() const {
    return projection_mat_;
  }

  const Mat44 &projection_mat_inv() const {
    return projection_mat_inv_;
  }

 private:
  Mat44 projection_mat_;
  Mat44 projection_mat_inv_;
};

class PerspectiveCamera : public Camera {
 public:
  PerspectiveCamera(const Vec3 &camera_position,
                    const Vec3 &lookat_position,
                    const Vec3 &up,
                    const FrustumParams &frustum_params);

  const Mat44 &projection_mat() const {
    return projection_mat_;
  }

  const Mat44 &projection_mat_inv() const {
    return projection_mat_inv_;
  }

 private:
  Mat44 projection_mat_;
  Mat44 projection_mat_inv_;
};
}
