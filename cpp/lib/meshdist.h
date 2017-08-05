//
// Created by daeyun on 4/28/17.
//

#ifndef MESHDIST_MESHDIST_H
#define MESHDIST_MESHDIST_H

#include <random>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <cmath>
#include <type_traits>
#include <iostream>
#include <boost/optional.hpp>

#ifndef NDEBUG
#define USE_OMP 0
#define OMP_NUM_THREADS 0
#else
#define USE_OMP 1
#endif

#include <omp.h>

namespace meshdist {
typedef float Float;

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

//#define LOG(msg) std::cout << (#msg) << ": " << msg << std::endl

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define LIKELY(x) (!!(x))
#endif

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define Expects(cond) (LIKELY(cond) ? static_cast<void>(0) : \
  throw std::runtime_error(__FILE__ ":" STRINGIFY(__LINE__) ":: Condition " #cond " failed."))

constexpr Float kEpsilon = 1e-8;

enum class Part { A, B, C, AB, BC, CA, ABC };

long MilliSecondsSinceEpoch();

long NanoSecondsSinceEpoch();

std::default_random_engine &RandomEngine();
Float Random();

class Triangle {
 public:
  Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c) : a(a), b(b), c(c) {
    OnTransformation();
  }

  void ApplyRt(const Mat34 &M) {
    auto R = M.leftCols<3>();
    auto t = M.col(3);
    a = (R * a + t).eval();
    b = (R * b + t).eval();
    c = (R * c + t).eval();
    OnTransformation();
  }

  void ApplyR(const Mat33 &M) {
    a = (M * a).eval();
    b = (M * b).eval();
    c = (M * c).eval();
    OnTransformation();
  }

  void Translate(const Vec3 &dxyz) {
    a += dxyz;
    b += dxyz;
    c += dxyz;
    OnTransformation();
  }

  void Print() const {
    std::cout << a.transpose() << ",      "
              << b.transpose() << ",      "
              << c.transpose() << std::endl;
  }

  Vec3 SamplePoint() const {
    Vec2 v12{Random(), Random()};
    if (v12.sum() > 1) {
      v12 = 1 - v12.array();
    }
    return a + (v12(0) * ab_) + (v12(1) * ac_);
  }

  Float Area() const {
    return ab_.cross(ac_).norm() * 0.5f;
  }

  void OnTransformation() {
    assert(a.allFinite());
    assert(b.allFinite());
    assert(c.allFinite());

    ab_ = b - a;
    ac_ = c - a;
  }

  const Vec3 &operator[](int i) {
    switch (i) {
      case 0:
        return a;
      case 1:
        return b;
      default:
        return c;
    }
  }

  Vec3 a, b, c;

 private:
  Vec3 ab_, ac_;
};

Points3 ApplyMat34(const Mat34 &M, const Points3 &points);

Points3 ApplyMat34Normal(const Mat34 &M, const Points3 &normals);

Mat33 AxisRotation(double angle, const Vec3 &axis);

Mat34 AxisRotation(double angle, const Vec3 &origin, const Vec3 &direction);

// Counter-clockwise rotation.
Mat22 Rotation2d(Float angle);

double DegreesToRadians(double degrees);

Float DistanceToLine(const Vec3 &origin, const Vec3 &direction, const Vec3 &point);

Mat33 ZRotation(Float angle);

void YZTransform(Triangle *verts, Mat34 *M = nullptr);

bool IsLeft(const Vec2 &p, const Vec2 &origin, const Vec2 &direction);

Vec2 CCRotate2d(const Vec2 &v);

std::vector<Part> FindClosestPart(Triangle &yz_triangle,
                                  const Points3 &points);

/**
 * @param triangles
 * @param points
 * @param all_distances [out] Row i of `all_distances` will contain distances from point i to all triangles.
 */
void DistanceToTriangles(const std::vector<Triangle> &triangles,
                         const Points3 &points,
                         Mat *all_distances);

void DistanceToTriangles2(const std::vector<Triangle> &triangles,
                          const Points3 &points,
                          Mat *all_distances);

void MinimumDistanceToTriangles(const std::vector<Triangle> &triangles,
                                const Points3 &points,
                                Vec *squared_distances);

void SamplePointsOnTriangles(const std::vector<Triangle> &triangles, float density, Points3 *points);

float MeshToMeshDistanceOneDirection(const std::vector<Triangle> &a,
                                     const std::vector<Triangle> &b,
                                     float sampling_density);

float MeshToMeshDistanceOneDirection2(const std::vector<Triangle> &a,
                                      const std::vector<Triangle> &b,
                                      float sampling_density);

float MeshToMeshDistanceOneDirection3(const std::vector<Triangle> &from,
                                      const std::vector<Triangle> &to,
                                      float sampling_density);

float MeshToMeshDistance(const std::vector<Triangle> &a,
                         const std::vector<Triangle> &b);

}

#endif //MESHDIST_MESHDIST_H
