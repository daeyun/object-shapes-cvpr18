//
// Created by daeyun on 5/10/17.
//

#include "meshdist.h"

#include <Eigen/Dense>
#include <type_traits>
#include <chrono>
#include <glog/logging.h>

#include "cpp/lib/common.h"
#include "cpp/lib/random_utils.h"
#include "benchmark.h"

namespace meshdist_cgal {

typedef CGAL::Simple_cartesian<float> K;
typedef std::list<K::Triangle_3>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

void Triangle::ApplyRt(const Mat34 &M) {
  auto R = M.leftCols<3>();
  auto t = M.col(3);
  a = (R * a + t).eval();
  b = (R * b + t).eval();
  c = (R * c + t).eval();
  OnTransformation();
}

void Triangle::ApplyR(const Mat33 &M) {
  a = (M * a).eval();
  b = (M * b).eval();
  c = (M * c).eval();
  OnTransformation();
}

void Triangle::Translate(const Vec3 &dxyz) {
  a += dxyz;
  b += dxyz;
  c += dxyz;
  OnTransformation();
}

void Triangle::Print() const {
  std::cout << a.transpose() << ",      "
            << b.transpose() << ",      "
            << c.transpose() << std::endl;
}

Vec3 Triangle::SamplePoint() const {
  Vec2 v12{mvshape::Random::Rand(), mvshape::Random::Rand()};
  if (v12.sum() > 1) {
    v12 = 1 - v12.array();
  }
  return a + (v12(0) * ab_) + (v12(1) * ac_);
}

double Triangle::Area() const {
  return ab_.cross(ac_).norm() * 0.5f;
}

void Triangle::OnTransformation() {
  assert(a.allFinite());
  assert(b.allFinite());
  assert(c.allFinite());

  ab_ = b - a;
  ac_ = c - a;
}

void SamplePointsOnTriangles(const std::vector<Triangle> &triangles, float density, Points3d *points) {
  std::vector<double> areas;
  areas.reserve(triangles.size());
  for (auto &&triangle : triangles) {
    areas.push_back(triangle.Area());
  }

  double surface_area = std::accumulate(areas.begin(), areas.end(), static_cast<double>(0));
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
      ++local_counts(distribution(mvshape::Random::Engine()), i_thread);
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

  Eigen::PermutationMatrix<Dynamic, Dynamic> perm(points->cols());
  perm.setIdentity();
  std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), mvshape::Random::Engine());
  *points *= perm; // permute columns
}

float MeshToMeshDistanceOneDirection(const std::vector<Triangle> &from,
                                     const std::vector<Triangle> &to,
                                     float sampling_density) {

  Points3d points;

  SamplePointsOnTriangles(from, sampling_density, &points);

  std::list<K::Triangle_3> triangle_list;
  for (const auto &triangle : to) {
    triangle_list.emplace_back(K::Point_3{triangle.a[0], triangle.a[1], triangle.a[2]},
                               K::Point_3{triangle.b[0], triangle.b[1], triangle.b[2]},
                               K::Point_3{triangle.c[0], triangle.c[1], triangle.c[2]});
  }
  std::vector<K::Point_3> point_list;
  for (int i = 0; i < points.cols(); ++i) {
    point_list.emplace_back(points(0, i), points(1, i), points(2, i));
  }

  int num_triangles = static_cast<int>(to.size());
  int num_points = static_cast<int>(points.cols());

  DLOG(INFO) << "Computing minimum distances from " << num_points
             << " points to " << num_triangles << " triangles.";

  auto start = mvshape::TimeSinceEpoch<std::milli>();
  Tree tree(triangle_list.begin(), triangle_list.end());
  tree.build();
  tree.accelerate_distance_queries();
  DLOG(INFO) << "Time elapsed for building tree (CGAL): " << mvshape::TimeSinceEpoch<std::milli>() - start;

  float distance_sum = 0;

#pragma omp parallel for if(USE_OMP) reduction(+:distance_sum) schedule(static)
  for (int i = 0; i < point_list.size(); ++i) {
    float dist = tree.squared_distance(point_list[i]);
    distance_sum += dist;
  }

  DLOG(INFO) << "distance: " << distance_sum;
  float rms = static_cast<float>(std::sqrt(distance_sum / static_cast<double>(point_list.size())));
  DLOG(INFO) << "RMS: " << rms;
  auto elapsed = mvshape::TimeSinceEpoch<std::milli>() - start;
  DLOG(INFO) << "Time elapsed (CGAL): " << elapsed << " ms";

  return rms;
}

float MeshToMeshDistance(const std::vector<Triangle> &a, const std::vector<Triangle> &b) {
  auto start = mvshape::TimeSinceEpoch<std::milli>();

  constexpr int kSamplingDensity = 300;

  float d1 = meshdist_cgal::MeshToMeshDistanceOneDirection(a, b, kSamplingDensity);
  float d2 = meshdist_cgal::MeshToMeshDistanceOneDirection(b, a, kSamplingDensity);

  auto elapsed = mvshape::TimeSinceEpoch<std::milli>() - start;
  DLOG(INFO) << "Time elapsed (MeshToMeshDistance): " << elapsed << " ms";
  DLOG(INFO) << d1 << ", " << d2;
  return static_cast<float>((d1 + d2) * 0.5);
}

}
