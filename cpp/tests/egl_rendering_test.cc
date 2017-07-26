//
// Created by daeyun on 6/14/17.
//

#include "gtest/gtest.h"
#include "glog/logging.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include "file_io.h"
#include "egl_rendering.h"
#include "resources.h"

using namespace mvshape;

TEST(Rendering, RenderMesh) {
  auto filename = Resources::ResourcePath("objects/1b8c83552010440d490ad276cd2af3a4/model.obj");
  auto triangles = FileIO::ReadTriangles(filename);

  Rendering::RendererConfig render_config{
      .width = 256,
      .height = 256,
  };

  Rendering::ShapeRenderer renderer(render_config);

  renderer.SetTriangleVertices(triangles);

  mvshape::FrustumParams frustum;
  frustum.near = 0.1;
  frustum.far = 10;

  mvshape::OrthographicCamera camera0({1, 0, 0}, {0, 0, 0}, {0, 1, 0}, frustum);
  mvshape::OrthographicCamera camera1({-1, 0, 0}, {0, 0, 0}, {0, 1, 0}, frustum);

  vector < vector < unique_ptr < cv::Mat >> > out_frames;
  renderer.Render(vector < mvshape::Camera * > {&camera0, &camera1}, &out_frames);

  EXPECT_EQ(2, out_frames.size());

  auto test_rendering_output = [&](int frame, int stream) {
    auto image = *out_frames[frame][stream];
    float *image_ptr = reinterpret_cast<float *>(image.data);
    auto image_floats = std::vector<float>(image_ptr, image_ptr + image.total() * image.elemSize() / sizeof(float));

    ASSERT_LT(0, *std::max_element(image_floats.begin(), image_floats.end()));

    std::sort(image_floats.begin(), image_floats.end());
    auto unique_count = std::unique(image_floats.begin(), image_floats.end()) - image_floats.begin();

    EXPECT_LT(500, unique_count);
  };

  test_rendering_output(0, 0);
  test_rendering_output(0, 1);
  test_rendering_output(1, 0);
  test_rendering_output(1, 1);
}

TEST(Rendering, RenderMeshDifferentSizesMultipleRenderers) {
  auto triangles = FileIO::ReadTriangles(Resources::ResourcePath("objects/torus.ply"));

  Rendering::RendererConfig render_config2{
      .width = 32,
      .height = 32,
  };
  Rendering::ShapeRenderer renderer2(render_config2);
  renderer2.SetTriangleVertices(triangles);

  Rendering::RendererConfig render_config{
      .width = 256,
      .height = 256,
  };
  Rendering::ShapeRenderer renderer(render_config);
  renderer.SetTriangleVertices(triangles);

  mvshape::FrustumParams frustum;
  frustum.near = 0.1;
  frustum.far = 10;

  mvshape::OrthographicCamera camera0({1, 0, 0}, {0, 0, 0}, {0, 1, 0}, frustum);
  mvshape::OrthographicCamera camera1({-1, 0, 0}, {0, 0, 0}, {0, 1, 0}, frustum);

  vector < vector < unique_ptr < cv::Mat >> > out_frames, out_frames2;
  renderer.Render(vector < mvshape::Camera * > {&camera0, &camera1}, &out_frames);
  renderer2.Render(vector < mvshape::Camera * > {&camera0, &camera1}, &out_frames2);

  EXPECT_EQ(2, out_frames.size());

  auto test_rendering_output = [&](int frame, int stream) {
    auto image = (stream == 0) ? *out_frames2[frame][0] : *out_frames[frame][1];
    float *image_ptr = reinterpret_cast<float *>(image.data);
    auto image_floats = std::vector<float>(image_ptr, image_ptr + image.total() * image.elemSize() / sizeof(float));

    ASSERT_LT(0, *std::max_element(image_floats.begin(), image_floats.end()));

    std::sort(image_floats.begin(), image_floats.end());
    auto unique_count = std::unique(image_floats.begin(), image_floats.end()) - image_floats.begin();
    LOG(INFO) << unique_count;

    EXPECT_LT(50, unique_count);
  };

  test_rendering_output(0, 0);
  test_rendering_output(0, 1);
  test_rendering_output(1, 0);
  test_rendering_output(1, 1);
}

// @TODO(daeyun): move this somewhere else.
TEST(DepthRecon, TorusPointCloudFromNormals) {
  std::string compressed = FileIO::ReadBytes(Resources::ResourcePath("rendering/torus_normals_0.bin"));
  const int *compressed_ptr = reinterpret_cast<const int *>(compressed.data());
  const int n = *compressed_ptr;
  const int height = *(compressed_ptr + 1);
  const int width = *(compressed_ptr + 2);
  EXPECT_EQ(3, *(compressed_ptr + 3));

  compressed_ptr += n + 1;

  std::string decompressed;
  FileIO::DecompressBytes(compressed_ptr, &decompressed);

  std::vector<float> xs(height * width);
  std::vector<float> ys(height * width);
  std::vector<float> zs(height * width);

  const float *data_ptr = reinterpret_cast<const float *>(decompressed.data());
  for (int i = 0; i < height * width; ++i) {
    xs[i] = *(data_ptr + i * 3);
    ys[i] = *(data_ptr + i * 3 + 1);
    zs[i] = *(data_ptr + i * 3 + 2);
  }

  for (int i = 0; i < height * width; ++i) {
    if (xs[i] == 0) {
      EXPECT_EQ(0, ys[i]);
      EXPECT_EQ(0, zs[i]);
    } else if (zs[i] != 0) {
      EXPECT_NE(0, xs[i]);
      EXPECT_NE(0, zs[i]);
    }
  }

  EXPECT_EQ(256, height);
  EXPECT_EQ(256, width);

  double s = 2 / 256.0;

  auto sub2ind = [&](int i, int j) {
    return i * width + j;
  };

  vector<int> valid_sub2ind;
  vector <std::pair<int, int>> valid_coords;
  int valid_count = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (zs[sub2ind(i, j)] != 0) {
        valid_sub2ind.push_back(valid_count);
        valid_coords.emplace_back(i, j);
        valid_count++;
      } else {
        valid_sub2ind.push_back(-1);
      }
    }
  }

  auto equation_sub2ind = [&](int i, int j) -> int {
    return valid_sub2ind[sub2ind(i, j)];
  };

  vector <Vec3> points;

  vector <std::tuple<int, int, double>> sparse;
  vector<double> rhs;
  int row_count = 0;

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      int equation_ind = equation_sub2ind(i, j);
      if (equation_ind < 0) {
        continue;
      }
      // Right
      int right_equation_ind = equation_sub2ind(i, j + 1);
      if (right_equation_ind >= 0) {
        int ind = sub2ind(i, j + 1);
        double dz = -xs[ind] / zs[ind] * s;

        sparse.emplace_back(row_count, equation_ind, -1);
        sparse.emplace_back(row_count, right_equation_ind, 1);
        rhs.push_back(dz);
        row_count++;
      }

      // Left
      int left_equation_ind = equation_sub2ind(i, j - 1);
      if (left_equation_ind >= 0) {
        int ind = sub2ind(i, j - 1);
        double dz = -xs[ind] / zs[ind] * s;

        sparse.emplace_back(row_count, left_equation_ind, -1);
        sparse.emplace_back(row_count, equation_ind, 1);
        rhs.push_back(dz);
        row_count++;
      }

      // Down
      int down_equation_ind = equation_sub2ind(i + 1, j);
      if (down_equation_ind >= 0) {
        int ind = sub2ind(i + 1, j);
        double dz = -ys[ind] / zs[ind] * s;

        sparse.emplace_back(row_count, down_equation_ind, -1);
        sparse.emplace_back(row_count, equation_ind, 1);
        rhs.push_back(dz);
        row_count++;
      }

      // Up
      int up_equation_ind = equation_sub2ind(i - 1, j);
      if (up_equation_ind >= 0) {
        int ind = sub2ind(i - 1, j);
        double dz = -ys[ind] / zs[ind] * s;

        sparse.emplace_back(row_count, equation_ind, -1);
        sparse.emplace_back(row_count, up_equation_ind, 1);
        rhs.push_back(dz);
        row_count++;
      }
    }
  }

  Eigen::VectorXd b(rhs.size());
  for (int k = 0; k < rhs.size(); ++k) {
    b(k) = rhs[k];
  }

  Eigen::VectorXd x(valid_count);
//  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
//  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
//  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
//  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  Eigen::SparseMatrix<double> rows(rhs.size(), valid_count);
  for (const auto &item : sparse) {
    rows.insert(std::get<0>(item), std::get<1>(item)) = std::get<2>(item);
  }
//  rows.makeCompressed();
  LOG(INFO) << rows.cols() << ", " << rows.rows();
  LOG(INFO) << b.cols() << ", " << b.rows();
  solver.compute(rows);
//  if (solver.info() != Success) {
//    LOG(ERROR) << "decomposition failed";
//  }
  x = solver.solve(b);
//  if (solver.info() != Success) {
//    LOG(ERROR) << "solving failed";
//  }

  Points3d out_pts(3, x.rows());
  int ii = 0;
  for (const auto &coord : valid_coords) {
    out_pts(0, ii) = (coord.first + 0.5) / 256.0 * 2;
    out_pts(1, ii) = (coord.second + 0.5) / 256.0 * 2;
    out_pts(2, ii) = x(ii);
    ii++;
  }

//  FileIO::SerializeMatrix("/home/daeyun/tmp/pts2.bin", out_pts.cast<float>());
}

TEST(SanityCheck, HelloWorld) {
//  RunDemo();
}
