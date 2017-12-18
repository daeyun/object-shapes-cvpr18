//
// Created by daeyun on 5/14/17.
//

#ifndef MESHDIST_MESHDIST_CGAL_H
#define MESHDIST_MESHDIST_CGAL_H

#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <omp.h>

#include "cpp/lib/common.h"

namespace meshdist_cgal {

class Triangle {
 public:
  Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c) : a(a), b(b), c(c) {
    OnTransformation();
  }

  void ApplyRt(const Mat34 &M);

  void ApplyR(const Mat33 &M);

  void Translate(const Vec3 &dxyz);

  void Print() const;

  Vec3 SamplePoint() const;

  double Area() const;

  void OnTransformation();

  const Vec3 &operator[](int i) {
    switch (i) {
      case 0:return a;
      case 1:return b;
      default:return c;
    }
  }

  Vec3 a, b, c;

 private:
  Vec3 ab_, ac_;
};

void SamplePointsOnTriangles(const std::vector<Triangle> &triangles, float density, Points3d *points);

float MeshToMeshDistanceOneDirection(const std::vector<Triangle> &from,
                                     const std::vector<Triangle> &to,
                                     float sampling_density);

float PointsToMeshDistanceOneDirection(const std::vector<std::array<float, 3>> &from,
                                       const std::vector<Triangle> &to);

float MeshToPointsDistanceOneDirection(const std::vector<Triangle> &from,
                                       const std::vector<std::array<float, 3>> &target_points,
                                       float sampling_density);

float MeshToMeshDistance(const std::vector<Triangle> &a, const std::vector<Triangle> &b);

}

#endif //MESHDIST_MESHDIST_CGAL_H
