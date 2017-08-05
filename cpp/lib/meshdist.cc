//
// Created by daeyun on 5/10/17.
//

#include "meshdist.h"

#include <Eigen/Dense>
#include <type_traits>
#include <glog/logging.h>

namespace meshdist {

using Eigen::Dynamic;
using Vec = Eigen::Matrix<Float, Dynamic, 1>;
using Vec2 = Eigen::Matrix<Float, 2, 1>;
using Vec3 = Eigen::Matrix<Float, 3, 1>;
using Vec4 = Eigen::Matrix<Float, 4, 1>;
using Mat = Eigen::Matrix<Float, Dynamic, Dynamic>;
using Mat44 = Eigen::Matrix<Float, 4, 4>;
using Mat34 = Eigen::Matrix<Float, 3, 4>;
using Mat33 = Eigen::Matrix<Float, 3, 3>;
using Mat22 = Eigen::Matrix<Float, 2, 2>;
using Points3 = Eigen::Matrix<Float, 3, Dynamic>;
using Points2d = Eigen::Matrix<Float, 2, Dynamic>;
using Points2i = Eigen::Matrix<int, 2, Dynamic>;

long MilliSecondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
      (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

long NanoSecondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

Float Random() {
  thread_local static std::uniform_real_distribution<Float> dist{0, 1};
  return dist(RandomEngine());
}

std::default_random_engine &RandomEngine() {
  thread_local static std::default_random_engine engine{std::random_device{}()};
  return engine;
}

Points3 ApplyMat34(const Mat34 &M, const Points3 &points) {
  // Faster than M*points.colwise().homogeneous()
  return (M.leftCols<3>() * points).colwise() + M.col(3);
}

Points3 ApplyMat34Normal(const Mat34 &M, const Points3 &normals) {
  return (M.leftCols<3>() * normals).array().rowwise()
      / ((-M.leftCols<3>().transpose() * normals).array() * normals.array()).colwise().sum();
}

Mat33 AxisRotation(double angle, const Vec3 &axis) {
  return Eigen::AngleAxis<Float>(angle, axis.normalized()).matrix();
}

Mat34 AxisRotation(double angle, const Vec3 &origin, const Vec3 &direction) {
  auto R = Eigen::AngleAxis<Float>(angle, direction.normalized());
  return (Eigen::Translation<Float, 3>(origin) * R * Eigen::Translation<Float, 3>(-origin)).matrix().topRows(3);
}

double DegreesToRadians(double degrees) {
  static const double kPi = M_PI / 180.0;
  return degrees * kPi;
}

Float DistanceToLine(const Vec3 &origin, const Vec3 &direction, const Vec3 &point) {
  return Eigen::ParametrizedLine<Float, 3>(origin, direction.normalized()).distance(point);
//  return direction.cast<double>().cross(point.cast<double>() - origin.cast<double>()).norm()
//      / direction.cast<double>().norm();
}

void YZTransform(Triangle *verts, Mat34 *M) {
  // 1. Translate so that A = (0, 0, 0).
  Vec3 translation = verts->a;
  verts->b.array() -= translation.array();
  verts->c.array() -= translation.array();
  verts->a = {0, 0, 0};

  // 2. Rotate so that AB aligns with the positive z axis.
  Vec3 ab = verts->b;
  Eigen::AngleAxis<Float> R1;

  // Angle between AB and z-axis.
  Float theta;
  if (ab.head<2>().isZero()) {
    // If ab is already on the z-axis
    if (ab(2) > kEpsilon) {
      // z is positive
      R1 = Eigen::AngleAxis<Float>::Identity();
    } else {
      // z is negative. Flip around y-axis.
      Vec3 rot_axis;
      rot_axis = Vec3::UnitY();
      theta = static_cast<Float>(M_PI);
      R1 = Eigen::AngleAxis<Float>(theta, rot_axis);
    }
  } else {
    ab.normalize();
    Vec3 rot_axis;
    rot_axis = ab.cross(Vec3::UnitZ());
    Float cross_norm = rot_axis.norm();

    // More stable than arccos.
    Float cos_theta = ab.dot(Vec3::UnitZ());
    theta = std::atan2(cross_norm, cos_theta);
    rot_axis.array() /= cross_norm;
    R1 = Eigen::AngleAxis<Float>(theta, rot_axis);
  }

  // 3. Rotate around z-axis so that the triangle is on the yz plane.
  // y will be positive.
  Vec2 cxy = (R1 * verts->c).topRows<2>();
  Float theta_z = std::atan2(cxy(0), cxy(1));

  Eigen::AngleAxis<Float> R2;
  if (std::abs(theta_z) < kEpsilon) {
    R2 = Eigen::AngleAxis<Float>::Identity();
  } else {
    R2 = Eigen::AngleAxis<Float>(theta_z, Vec3::UnitZ());
  }

  auto R = R2 * R1;

  verts->b = (R * verts->b).eval();
  verts->c = (R * verts->c).eval();
  verts->OnTransformation();

  if (M != nullptr) {
    M->topLeftCorner<3, 3>() = R.matrix();
    M->col(3).array() = R * -translation;
  }
}

bool IsLeft(const Vec2 &p, const Vec2 &origin, const Vec2 &direction) {
  const auto po = p - origin;
  auto d0 = po(0) * direction(1);
  auto d1 = po(1) * direction(0);
  return d0 - d1 < 0.0;
}

Vec2 CCRotate2d(const Vec2 &v) {
  return {-v[1], v[0]};
}

std::vector<Part> FindClosestPart(Triangle &yz_triangle, const Points3 &points) {
  auto p1 = yz_triangle.a.tail<2>();
  auto p2 = yz_triangle.b.tail<2>();
  auto p3 = yz_triangle.c.tail<2>();

  Vec2 d31 = p1 - p3;
  Vec2 d31_cc = CCRotate2d(d31);
  Vec2 d12 = p2 - p1;
  Vec2 d12_cc = CCRotate2d(d12);
  Vec2 d23 = p3 - p2;
  Vec2 d23_cc = CCRotate2d(d23);

  std::vector<Part> parts;
  const size_t n = static_cast<size_t>(points.cols());
  parts.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    Vec2 point_yz = points.col(i).tail<2>();

    bool s2 = IsLeft(point_yz, p1, d31_cc);
    bool s4 = IsLeft(point_yz, p1, d12_cc);
    if (!s2 && s4) {
      parts.push_back(Part::A);
      continue;
    }

    bool s5 = IsLeft(point_yz, p2, d12_cc);
    bool s7 = IsLeft(point_yz, p2, d23_cc);
    if (!s5 && s7) {
      parts.push_back(Part::B);
      continue;
    }

    bool s1 = IsLeft(point_yz, p3, d31_cc);
    bool s8 = IsLeft(point_yz, p3, d23_cc);
    if (!s8 && s1) {
      parts.push_back(Part::C);
      continue;
    }

    bool s3 = IsLeft(point_yz, p1, d12);
    if (s3 && !s4 && s5) {
      parts.push_back(Part::AB);
      continue;
    }

    bool s6 = IsLeft(point_yz, p2, d23);
    if (s6 && !s7 && s8) {
      parts.push_back(Part::BC);
      continue;
    }

    bool s0 = IsLeft(point_yz, p3, d31);
    if (s0 && !s1 && s2) {
      parts.push_back(Part::CA);
      continue;
    }

    parts.push_back(Part::ABC);
  }

  return parts;
}

void DistanceToTriangles(const std::vector<Triangle> &triangles, const Points3 &points, Mat *all_distances) {
  all_distances->resize(points.cols(), triangles.size());

  Expects(!all_distances->IsRowMajor);

  int num_triangles = static_cast<int>(triangles.size());
  int num_points = static_cast<int>(points.cols());

#pragma omp parallel for if(USE_OMP)
  for (int j = 0; j < num_triangles; ++j) {
    Mat34 T;
    Triangle yz_triangle = triangles[j];
    YZTransform(&yz_triangle, &T);

    Points3 Tpoints = ApplyMat34(T, points);
    auto parts = FindClosestPart(yz_triangle, Tpoints);

    Float sin_bc, cos_bc, sin_ca, cos_ca;
    bool first_sin_cos_bc = true;
    bool first_sin_cos_ca = true;

    for (int i = 0; i < num_points; ++i) {
      Part part = parts[i];
      Vec3 Tpoint = Tpoints.col(i).eval();

      Float dist;
      switch (part) {
        case Part::A:dist = Tpoint.norm();
          break;
        case Part::B:Tpoint(2) -= yz_triangle.b(2);
          dist = Tpoint.norm();
          break;
        case Part::C:Tpoint.bottomRows(2) -= yz_triangle.c.bottomRows(2);
          dist = Tpoint.norm();
          break;
        case Part::AB:dist = Tpoint.topRows(2).norm();
          break;
        case Part::BC: {
          if (first_sin_cos_bc) {
            Float theta_bc = -std::atan2(yz_triangle.c(2) - yz_triangle.b(2), yz_triangle.c(1));
            sin_bc = std::sin(theta_bc);
            cos_bc = std::cos(theta_bc);
            first_sin_cos_bc = false;
          }

          Float z;
          z = sin_bc * Tpoint(1) + cos_bc * (Tpoint(2) - yz_triangle.b(2));
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::CA: {
          if (first_sin_cos_ca) {
            Float theta_ca = -std::atan2(yz_triangle.c(2), yz_triangle.c(1));
            sin_ca = std::sin(theta_ca);
            cos_ca = std::cos(theta_ca);
            first_sin_cos_ca = false;
          }

          Float z;
          z = sin_ca * Tpoint(1) + cos_ca * Tpoint(2);
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::ABC:dist = std::abs(Tpoint(0));
          break;
      }
      all_distances->coeffRef(i, j) = dist;
    }
  }
}

void DistanceToTriangles2(const std::vector<Triangle> &triangles, const Points3 &points, Mat *all_distances) {
  all_distances->resize(points.cols(), triangles.size());

  Expects(!all_distances->IsRowMajor);

  int num_triangles = static_cast<int>(triangles.size());

#pragma omp parallel for if(USE_OMP)
  for (int j = 0; j < num_triangles; ++j) {
    Mat34 T;
    Triangle yz_triangle = triangles[j];
    YZTransform(&yz_triangle, &T);
    Points3 Tpoints = ApplyMat34(T, points);

    auto p1 = yz_triangle.a.tail<2>();
    auto p2 = yz_triangle.b.tail<2>();
    auto p3 = yz_triangle.c.tail<2>();

    Vec2 d31 = p1 - p3;
    Vec2 d31_cc = CCRotate2d(d31);
    Vec2 d12 = p2 - p1;
    Vec2 d12_cc = CCRotate2d(d12);
    Vec2 d23 = p3 - p2;
    Vec2 d23_cc = CCRotate2d(d23);

    const size_t n = static_cast<size_t>(Tpoints.cols());

    Float sin_bc, cos_bc, sin_ca, cos_ca;
    bool first_sin_cos_bc = true;
    bool first_sin_cos_ca = true;

    for (size_t i = 0; i < n; ++i) {
      Vec2 point_yz = Tpoints.col(i).tail<2>();

#if 0
      Part part;
    bool s2 = IsLeft(point_yz, p1, d31_cc);
    bool s4 = IsLeft(point_yz, p1, d12_cc);

    bool s5 = IsLeft(point_yz, p2, d12_cc);
    bool s7 = IsLeft(point_yz, p2, d23_cc);

    bool s1 = IsLeft(point_yz, p3, d31_cc);
    bool s8 = IsLeft(point_yz, p3, d23_cc);
    bool s3 = IsLeft(point_yz, p1, d12);
    bool s6 = IsLeft(point_yz, p2, d23);
    bool s0 = IsLeft(point_yz, p3, d31);

    if (!s2 && s4) {
      part = Part::A;
    } else if (!s5 && s7) {
      part = Part::B;
    } else if (!s8 && s1) {
      part = Part::C;
    } else if (s3 && !s4 && s5) {
      part = Part::AB;
    } else if (s6 && !s7 && s8) {
      part = Part::BC;
    } else if (s0 && !s1 && s2) {
      part = Part::CA;
    } else {
      part = Part::ABC;
    }
#endif

#if 1
      const Part part = [&] {
        bool s2 = IsLeft(point_yz, p1, d31_cc);
        bool s4 = IsLeft(point_yz, p1, d12_cc);
        if (!s2 && s4) {
          return Part::A;
        }

        bool s5 = IsLeft(point_yz, p2, d12_cc);
        bool s7 = IsLeft(point_yz, p2, d23_cc);
        if (!s5 && s7) {
          return Part::B;
        }

        bool s1 = IsLeft(point_yz, p3, d31_cc);
        bool s8 = IsLeft(point_yz, p3, d23_cc);
        if (!s8 && s1) {
          return Part::C;
        }

        bool s3 = IsLeft(point_yz, p1, d12);
        if (s3 && !s4 && s5) {
          return Part::AB;
        }

        bool s6 = IsLeft(point_yz, p2, d23);
        if (s6 && !s7 && s8) {
          return Part::BC;
        }

        bool s0 = IsLeft(point_yz, p3, d31);
        if (s0 && !s1 && s2) {
          return Part::CA;
        }

        return Part::ABC;
      }();
#endif

      Vec3 Tpoint = Tpoints.col(i);

      Float dist;
      switch (part) {
        case Part::A:dist = Tpoint.norm();
          break;
        case Part::B:Tpoint(2) -= p2(1);
          dist = Tpoint.norm();
          break;
        case Part::C:Tpoint.bottomRows(2) -= p3;
          dist = Tpoint.norm();
          break;
        case Part::AB:dist = Tpoint.topRows(2).norm();
          break;
        case Part::BC: {
          if (first_sin_cos_bc) {
            // Same as arctan(c_z - b_z, c_x).
            Float theta_bc = -std::atan2(d23(1), p3(0));
            sin_bc = std::sin(theta_bc);
            cos_bc = std::cos(theta_bc);
            first_sin_cos_bc = false;
          }

          Float z = sin_bc * Tpoint(1) + cos_bc * (Tpoint(2) - p2(1));
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::CA: {
          if (first_sin_cos_ca) {
            Float theta_ca = -std::atan2(p3(1), p3(0));
            sin_ca = std::sin(theta_ca);
            cos_ca = std::cos(theta_ca);
            first_sin_cos_ca = false;
          }

          Float z = sin_ca * Tpoint(1) + cos_ca * Tpoint(2);
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::ABC:dist = std::abs(Tpoint(0));
          break;
      }
      all_distances->coeffRef(i, j) = dist;
    }
  }
}

void MinimumDistanceToTriangles(const std::vector<Triangle> &triangles,
                                const Points3 &points,
                                Vec *squared_distances) {
  const int num_triangles = static_cast<int>(triangles.size());
  const int num_points = static_cast<int>(points.cols());

  LOG(INFO) << "Computing minimum distances from " << num_points
            << " points to " << num_triangles << " triangles.";

  std::vector<std::unique_ptr<Float[]>> all_local_distances;

#pragma omp parallel if (USE_OMP)
  {
    std::unique_ptr<Float[]> local_distances(new Float[num_points]);
    std::fill(local_distances.get(), local_distances.get() + num_points,
              std::numeric_limits<Float>::max());

#pragma omp for
    for (int j = 0; j < num_triangles; ++j) {
      Mat34 T;
      Triangle yz_triangle = triangles[j];
      YZTransform(&yz_triangle, &T);

      auto p1 = yz_triangle.a.tail<2>();
      auto p2 = yz_triangle.b.tail<2>();
      auto p3 = yz_triangle.c.tail<2>();

      Vec2 d31 = p1 - p3;
      Vec2 d31_cc = CCRotate2d(d31);
      Vec2 d12 = p2 - p1;
      Vec2 d12_cc = CCRotate2d(d12);
      Vec2 d23 = p3 - p2;
      Vec2 d23_cc = CCRotate2d(d23);

      Float sin_bc, cos_bc, sin_ca, cos_ca;
      bool first_sin_cos_bc = true;
      bool first_sin_cos_ca = true;

      for (size_t i = 0; i < num_points; ++i) {
        auto p = points.col(i);

        Vec3 Tpoint;

        Tpoint(0) = T.topLeftCorner<1, 3>() * p + T(0, 3);
        Float xdist = Tpoint(0) * Tpoint(0);

        Float current_minimum = local_distances[i];
        if (xdist > current_minimum) {
          // A heuristic to skip if the lower bound is too big.
          continue;
        }

        Tpoint.tail<2>() = T.bottomLeftCorner<2, 3>() * p + T.bottomRightCorner<2, 1>();
        auto point_yz = Tpoint.tail<2>();

        auto yz_max = std::max(-point_yz(0), std::abs(point_yz(1)));
        if (point_yz(0) < 0 && yz_max * yz_max > current_minimum) {
          continue;
        }

        const Part part = [&] {
          bool s2 = IsLeft(point_yz, p1, d31_cc);
          bool s4 = IsLeft(point_yz, p1, d12_cc);
          if (!s2 && s4) {
            return Part::A;
          }

          bool s5 = IsLeft(point_yz, p2, d12_cc);
          bool s7 = IsLeft(point_yz, p2, d23_cc);
          if (!s5 && s7) {
            return Part::B;
          }

          bool s1 = IsLeft(point_yz, p3, d31_cc);
          bool s8 = IsLeft(point_yz, p3, d23_cc);
          if (!s8 && s1) {
            return Part::C;
          }

          bool s3 = IsLeft(point_yz, p1, d12);
          if (s3 && !s4 && s5) {
            return Part::AB;
          }

          bool s6 = IsLeft(point_yz, p2, d23);
          if (s6 && !s7 && s8) {
            return Part::BC;
          }

          bool s0 = IsLeft(point_yz, p3, d31);
          if (s0 && !s1 && s2) {
            return Part::CA;
          }

          return Part::ABC;
        }();

        Float dist;
        switch (part) {
          case Part::A:dist = Tpoint.squaredNorm();
            break;
          case Part::B:Tpoint(2) -= p2(1);
            dist = Tpoint.squaredNorm();
            break;
          case Part::C:Tpoint.bottomRows(2) -= p3;
            dist = Tpoint.squaredNorm();
            break;
          case Part::AB:dist = Tpoint.topRows(2).squaredNorm();
            break;
          case Part::BC: {
            if (first_sin_cos_bc) {
              // Same as arctan(c_z - b_z, c_x).
              Float theta_bc = -std::atan2(d23(1), p3(0));
              sin_bc = std::sin(theta_bc);
              cos_bc = std::cos(theta_bc);
              first_sin_cos_bc = false;
            }
            Float z = sin_bc * Tpoint(1) + cos_bc * (Tpoint(2) - p2(1));
            dist = z * z + Tpoint(0) * Tpoint(0);
            break;
          }
          case Part::CA: {
            if (first_sin_cos_ca) {
              Float theta_ca = -std::atan2(p3(1), p3(0));
              sin_ca = std::sin(theta_ca);
              cos_ca = std::cos(theta_ca);
              first_sin_cos_ca = false;
            }

            Float z = sin_ca * Tpoint(1) + cos_ca * Tpoint(2);
            dist = z * z + Tpoint(0) * Tpoint(0);
            break;
          }
          case Part::ABC:dist = xdist;
            break;
        }

        if (dist < current_minimum) {
          local_distances[i] = dist;
        }
      }
    }
#pragma omp critical
    all_local_distances.push_back(std::move(local_distances));
  }

  squared_distances->resize(num_points);
  for (int k = 0; k < all_local_distances.size(); ++k) {
    for (size_t i = 0; i < num_points; ++i) {
      Float value = all_local_distances[k][i];
      if (k == 0 || value < squared_distances->coeff(i)) {
        squared_distances->coeffRef(i) = value;
      }
    }
  }
}

void SamplePointsOnTriangles(const std::vector<Triangle> &triangles, float density, Points3 *points) {
  std::vector<Float> areas;
  areas.reserve(triangles.size());
  for (auto &&triangle : triangles) {
    areas.push_back(triangle.Area());
  }

  Float surface_area = std::accumulate(areas.begin(), areas.end(), static_cast<Float>(0));
  int num_samples = static_cast<int>(surface_area * density);
  Expects(num_samples > 0);

  std::discrete_distribution<> distribution(std::begin(areas), std::end(areas));

  Eigen::Matrix<int, Dynamic, 1> counts;

  Eigen::Matrix<int, Dynamic, Dynamic> local_counts;

#pragma omp parallel if (USE_OMP && num_samples > 1e6)
  {
    const int num_threads = omp_get_num_threads();
    const int i_thread = omp_get_thread_num();

#pragma omp single
    {
      local_counts.resize(triangles.size(), num_threads);
      local_counts.fill(0);
    }

#pragma omp for
    for (int i = 0; i < num_samples; ++i) {
      ++local_counts(distribution(RandomEngine()), i_thread);
    }
  }

  counts = local_counts.rowwise().sum();

  points->resize(3, num_samples);
  int k = 0;
  for (int i = 0; i < triangles.size(); ++i) {
    for (int j = 0; j < counts(i); ++j) {
      points->col(k) = triangles[i].SamplePoint();
      ++k;
    }
  }
  Expects(k == num_samples);
}

float MeshToMeshDistanceOneDirection(const std::vector<Triangle> &a,
                                     const std::vector<Triangle> &b,
                                     float sampling_density) {
  Points3 points;
  Mat all_distances;

  SamplePointsOnTriangles(a, sampling_density, &points);

  auto start = MilliSecondsSinceEpoch();
  DistanceToTriangles2(b, points, &all_distances);
  auto elapsed = MilliSecondsSinceEpoch() - start;

  int num_points = static_cast<int>(points.cols());

  float rms = 0;
#pragma omp parallel for reduction(+:rms) if (USE_OMP)
  for (int i = 0; i < num_points; ++i) {
    float min_distance = all_distances.row(i).minCoeff();
    rms += min_distance * min_distance;
  }
  rms /= static_cast<float>(num_points);
  rms = std::sqrt(rms);
  return rms;
}

float MeshToMeshDistanceOneDirection2(const std::vector<Triangle> &a,
                                      const std::vector<Triangle> &b,
                                      float sampling_density) {
  Points3 points;

  SamplePointsOnTriangles(a, sampling_density, &points);

  const int num_triangles = static_cast<int>(b.size());
  const int num_points = static_cast<int>(points.cols());

  auto start = MilliSecondsSinceEpoch();

  Mat all_distances;
  const int num_threads = omp_get_max_threads();
  all_distances.resize(points.cols(), num_threads);
  all_distances.fill(std::numeric_limits<Float>::max());

  Expects(!all_distances.IsRowMajor);

#pragma omp parallel for num_threads(num_threads) if (USE_OMP)
  for (int j = 0; j < num_triangles; ++j) {
    Mat34 T;
    Triangle yz_triangle = b[j];
    YZTransform(&yz_triangle, &T);
    Points3 Tpoints = ApplyMat34(T, points);

    auto p1 = yz_triangle.a.tail<2>();
    auto p2 = yz_triangle.b.tail<2>();
    auto p3 = yz_triangle.c.tail<2>();

    Vec2 d31 = p1 - p3;
    Vec2 d31_cc = CCRotate2d(d31);
    Vec2 d12 = p2 - p1;
    Vec2 d12_cc = CCRotate2d(d12);
    Vec2 d23 = p3 - p2;
    Vec2 d23_cc = CCRotate2d(d23);

    Float sin_bc, cos_bc, sin_ca, cos_ca;
    bool first_sin_cos_bc = true;
    bool first_sin_cos_ca = true;

    for (int i = 0; i < num_points; ++i) {
      Vec3 Tpoint = Tpoints.col(i);
      auto point_yz = Tpoint.tail<2>();

      const Part part = [&] {
        bool s2 = IsLeft(point_yz, p1, d31_cc);
        bool s4 = IsLeft(point_yz, p1, d12_cc);
        if (!s2 && s4) {
          return Part::A;
        }

        bool s5 = IsLeft(point_yz, p2, d12_cc);
        bool s7 = IsLeft(point_yz, p2, d23_cc);
        if (!s5 && s7) {
          return Part::B;
        }

        bool s1 = IsLeft(point_yz, p3, d31_cc);
        bool s8 = IsLeft(point_yz, p3, d23_cc);
        if (!s8 && s1) {
          return Part::C;
        }

        bool s3 = IsLeft(point_yz, p1, d12);
        if (s3 && !s4 && s5) {
          return Part::AB;
        }

        bool s6 = IsLeft(point_yz, p2, d23);
        if (s6 && !s7 && s8) {
          return Part::BC;
        }

        bool s0 = IsLeft(point_yz, p3, d31);
        if (s0 && !s1 && s2) {
          return Part::CA;
        }

        return Part::ABC;
      }();

      Float dist;
      switch (part) {
        case Part::A:dist = Tpoint.norm();
          break;
        case Part::B:Tpoint(2) -= p2(1);
          dist = Tpoint.norm();
          break;
        case Part::C:Tpoint.bottomRows(2) -= p3;
          dist = Tpoint.norm();
          break;
        case Part::AB:dist = Tpoint.topRows(2).norm();
          break;
        case Part::BC: {
          if (first_sin_cos_bc) {
            // Same as -arctan(c_z - b_z, c_x).
            Float theta_bc = -std::atan2(d23(1), p3(0));
            sin_bc = std::sin(theta_bc);
            cos_bc = std::cos(theta_bc);
            first_sin_cos_bc = false;
          }
          Float z = sin_bc * Tpoint(1) + cos_bc * (Tpoint(2) - p2(1));
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::CA: {
          if (first_sin_cos_ca) {
            Float theta_ca = -std::atan2(p3(1), p3(0));
            sin_ca = std::sin(theta_ca);
            cos_ca = std::cos(theta_ca);
            first_sin_cos_ca = false;
          }

          Float z = sin_ca * Tpoint(1) + cos_ca * Tpoint(2);
          dist = std::sqrt(z * z + Tpoint(0) * Tpoint(0));
          break;
        }
        case Part::ABC:dist = std::abs(Tpoint(0));
          break;
      }

      const int thread_num = omp_get_thread_num();
      Float val = all_distances(i, thread_num);
      if (dist < val) {
        all_distances(i, thread_num) = dist;
      }
    }
  }

  auto rms = std::sqrt(all_distances.rowwise().minCoeff().squaredNorm() / static_cast<Float>(num_points));
  auto elapsed = MilliSecondsSinceEpoch() - start;
  return rms;
}

float MeshToMeshDistanceOneDirection3(const std::vector<Triangle> &from,
                                      const std::vector<Triangle> &to,
                                      float sampling_density) {

  Points3 points;

  SamplePointsOnTriangles(from, sampling_density, &points);

  int num_triangles = to.size();
  int num_points = points.cols();

  auto start = MilliSecondsSinceEpoch();
  const int n = static_cast<int>(points.cols());

  Vec minimum_distances;
  MinimumDistanceToTriangles(to, points, &minimum_distances);

  auto elapsed = MilliSecondsSinceEpoch() - start;

  auto rms = std::sqrt(minimum_distances.sum() / static_cast<Float>(n));
  LOG(INFO) << "RMS: " << rms;

  return rms;
}

float MeshToMeshDistance(const std::vector<Triangle> &a, const std::vector<Triangle> &b) {
  auto start = MilliSecondsSinceEpoch();

  float d1 = MeshToMeshDistanceOneDirection3(a, b, 200000);
  float d2 = MeshToMeshDistanceOneDirection3(b, a, 200000);

  auto elapsed = MilliSecondsSinceEpoch() - start;
  LOG(INFO) << "Time elapsed (MeshToMeshDistance): " << elapsed << " ms";
  return (d1 + d2) * 0.5;
}
}
