//
// Created by daeyun on 8/4/17.
//

#define GSL_THROW_ON_CONTRACT_VIOLATION

#include "fssr_utils.h"

#include <iostream>
#include <fstream>
#include <omp.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gsl/gsl_assert>
#include <third_party/repos/mve/libs/mve/view.h>
#include <third_party/repos/mve/libs/mve/depthmap.h>
#include <third_party/repos/mve/libs/util/file_system.h>
#include <third_party/repos/mve/libs/mve/mesh_io.h>
#include <third_party/repos/mve/libs/mve/mesh_io_ply.h>
#include <third_party/repos/mve/libs/mve/mesh_tools.h>
#include <third_party/repos/mve/libs/mve/mesh_info.h>

#include "cpp/lib/data_io.h"
#include "cpp/lib/database.h"
#include "cpp/lib/flags.h"
#include "cpp/lib/common.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lockfree/queue.hpp>
#include <cpp/lib/file_io.h>
#include <third_party/repos/mve/libs/mve/scene.h>

#include "proto/dataset.pb.h"

DEFINE_double(scale_factor, 2.5, "'Radius' of MVS patch");

namespace mvshape {
namespace fssr {
namespace mv = mvshape_dataset;
namespace fs = boost::filesystem;
using boost::filesystem::path;
using boost::property_tree::ptree;

constexpr uint8_t kMveiHeader[] = {0x89, 0x4D, 0x56, 0x45, 0x5F, 0x49, 0x4D, 0x41, 0x47, 0x45, 0x0A};

template<typename T>
void WriteMvei(const string &filename, const vector<int> &shape, const vector<T> &data) {
  Expects(shape.size() == 3);
  size_t num_elements = static_cast<size_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  Ensures(data.size() == num_elements);

  std::ofstream outfile;
  outfile.open(filename);

  mvshape::FileIO::WriteBytes(static_cast<const void *>(kMveiHeader),
                              sizeof(kMveiHeader),
                              &outfile);
  auto type = mve::TypedImageBase<float>().get_type();

  // width, height, channels, type
  auto shape_type = std::vector<int32_t>{shape[1], shape[0], shape[2], type};
  mvshape::FileIO::WriteBytes<int32_t>(shape_type, &outfile);
  mvshape::FileIO::WriteBytes<T>(data, &outfile);

  outfile.close();
}

template void WriteMvei<float>(const string &filename, const vector<int> &shape, const vector<float> &data);

template<typename T>
void WriteMvei(const string &filename, const vector<int> &shape, const void *data, size_t size_bytes) {
  Expects(shape.size() == 3);
  FileIO::PrepareDir(path(filename).parent_path().string());
  std::ofstream outfile;
  outfile.open(filename);

  mvshape::FileIO::WriteBytes(static_cast<const void *>(kMveiHeader),
                              sizeof(kMveiHeader),
                              &outfile);
  auto type = mve::TypedImageBase<float>().get_type();

  // width, height, channels, type
  auto shape_type = std::vector<int32_t>{shape[1], shape[0], shape[2], type};
  mvshape::FileIO::WriteBytes<int32_t>(shape_type, &outfile);
  mvshape::FileIO::WriteBytes(data, size_bytes, &outfile);

  outfile.close();
}

template void WriteMvei<float>(const string &filename, const vector<int> &shape, const void *data, size_t size_bytes);

void PoissonScaleNormals(mve::TriangleMesh::ConfidenceList const &confs,
                         mve::TriangleMesh::NormalList *normals) {
  if (confs.size() != normals->size())
    throw std::invalid_argument("Invalid confidences or normals");
  for (std::size_t i = 0; i < confs.size(); ++i)
    normals->at(i) *= confs[i];
}

std::thread ReadSavedTensors(const string &tensor_dir, const vector<string> &subdir_names,
                             mvshape::concurrency::BatchQueue<FloatImages> *queue) {
  using boost::filesystem::path;
  auto tensor_reader = [tensor_dir, subdir_names, queue]() {
    // List files in subdirectories.
    map<string, vector<string>> subdir_filenames;
    // tensor_dir must have a directory named "index".
    Ensures(FileIO::Exists((path(tensor_dir) / "index").string()));
    subdir_filenames["index"] = FileIO::RegularFilesInDirectory((path(tensor_dir) / "index").string());
    size_t num_files = subdir_filenames["index"].size();
    Ensures(num_files > 0);
    for (const auto &subdir_name : subdir_names) {
      Expects(subdir_name != "index");
      // Sorted filenames.
      auto filenames = FileIO::RegularFilesInDirectory((path(tensor_dir) / subdir_name).string());
      Ensures(filenames.size() > 0);
      Ensures(num_files == filenames.size());
      subdir_filenames[subdir_name] = filenames;

      for (int i = 0; i < num_files; ++i) {
        const auto a = fs::basename(filenames[i]);
        const auto b = fs::basename(subdir_filenames["index"][i]);
        Ensures(a == b);
      }
    }

    for (int i = 0; i < num_files; ++i) {
      vector<int> index_shape;
      vector<int> index;
      FileIO::ReadTensorData<int>(subdir_filenames["index"][i], &index_shape, &index);
      Ensures(index_shape.size() == 1);
      const int batch_size = index_shape[0];
      Ensures(batch_size > 0);

      for (int b = 0; b < batch_size; ++b) {
        FloatImages view_images;
        view_images.index = index[b];

        for (const auto &item : subdir_filenames) {
          if (item.first == "index") {
            continue;
          }
          vector<int> shape;
          vector<float> float_data;
          FileIO::ReadTensorData<float>(item.second[i], &shape, &float_data);
          Ensures(shape.size() == 5);
          Ensures(shape[0] == batch_size);
          Ensures(shape[1] == 6);  // number of views
          Ensures(shape[2] > 0);  // height
          Ensures(shape[3] > 0);  // width
          Ensures(shape[4] == 1);  // channel

          const int num_views = shape[1];
          const int num_values_per_image = shape[2] * shape[3] * shape[4];

          for (int v = 0; v < num_views; ++v) {
            const int offset = num_values_per_image * (b * num_views + v);
            auto start = float_data.begin() + offset;
            auto end = start + num_values_per_image;
            mve::FloatImage::Ptr image = mve::FloatImage::create(shape[3], shape[2], shape[4]);
            Ensures(num_values_per_image == image->get_value_amount());
            std::copy(start, end, image->get_data_pointer());
            view_images.images[item.first].push_back(image);
          }
        }

        // emit view_images
        if (!queue->Enqueue(view_images)) {
          LOG(INFO) << "Queue closed.";
          return;
        }
        DLOG(INFO) << "Enqueue OK.";
      }
    }

    queue->Close();
    LOG(INFO) << "End of ReadSavedTensors";
  };

  return std::thread(tensor_reader);
}

void Depth2Mesh(const string &tensor_dir, const string &out_dir) {
  LOG(INFO) << "Depth2Mesh: " << tensor_dir;
  FileIO::PrepareDir(out_dir);
  mvshape::concurrency::BatchQueue<fssr::FloatImages> queue(8);
  auto thread = fssr::ReadSavedTensors(tensor_dir, {
      "placeholder_target_depth",
      "out_depth",
      "out_silhouette",
  }, &queue);

#pragma omp parallel if (USE_OMP)
  {

    while (true) {
      fssr::FloatImages images;
      auto dequeue_count = queue.Dequeue(&images, 1);

      if (dequeue_count == 0) {
        LOG(INFO) << "End of queue";
        break;
      }

      auto frustum = FrustumParams();
      frustum.near = Data::kNear;
      frustum.far = Data::kFar;

      Vec3 eye{15, 0, 0};
      Vec3 lookat{0, 0, 0};
      Vec3 up{0, 0, 1};

      // TODO
      if (tensor_dir.find("object_centered") != std::string::npos) {
        // no-op.
      } else if (tensor_dir.find("viewer_centered") != std::string::npos) {
        mv::Examples test_examples;
        // TODO
        Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/test.bin"), &test_examples);
        const auto &mv_depth = test_examples.examples(images.index).multiview_depth();
        eye = {mv_depth.eye(0), mv_depth.eye(1), mv_depth.eye(2)};
        lookat = {mv_depth.lookat(0), mv_depth.lookat(1), mv_depth.lookat(2)};
        up = {mv_depth.up(0), mv_depth.up(1), mv_depth.up(2)};
      } else {
//        throw std::runtime_error(tensor_dir);
        // no-op
        // OK for testing.
      }

      auto camera = OrthographicCamera(eye, lookat, up, frustum);

      Expects(6 == images.images["out_depth"].size());
      vector<unique_ptr<Camera>> cams;
      Data::SixViews(camera, &cams);

      auto merged_mesh = mve::TriangleMesh::create();

      for (int i = 0; i < cams.size(); ++i) {
        auto image = images.images["out_depth"][i];
        auto image_mask = images.images["out_silhouette"][i];

        Expects(image->get_value_amount() == image_mask->get_value_amount());

        for (int px = 0; px < image->get_value_amount(); ++px) {
          if (image_mask->at(px) <= 0.0) {
            image->at(px) = NAN;
          }
        }

        // TODO: 5.5
        std::transform(image->begin(), image->end(), image->begin(),
                       [](float num) -> float { return static_cast<float>(num + 5.5); });

        auto mesh = Depth2MeshSingle(image, *cams[i]);

        mve::geom::mesh_merge(mesh, merged_mesh);

        mve::geom::SavePLYOptions opts;
        opts.write_vertex_normals = true;
        opts.write_vertex_values = true;
        opts.write_vertex_confidences = true;
        opts.format_binary = true;
        std::ostringstream ply_name;
        ply_name << std::setw(4) << std::setfill('0') << images.index
                 << "/view_" << std::setw(4) << std::setfill('0') << i << ".ply";

        auto out_filename = (path(out_dir) / ply_name.str()).string();
        FileIO::PrepareDir(path(out_filename).parent_path().string());

        std::cout << "Writing " << out_filename << std::endl;
        mve::geom::save_ply_mesh(mesh, out_filename, opts);
      }
    }
#pragma omp barrier

  } // pragma omp parallel

  queue.Close();
  thread.join();
}

mve::TriangleMesh::Ptr Depth2MeshSingle(const mve::Image<float>::Ptr &image, const mvshape::Camera &camera) {
  Mat44 view_mat = camera.view_mat();

  Mat44 scale_mat = Mat44::Identity();

  // TODO
  scale_mat(0, 0) = 0.4;
  scale_mat(1, 1) = 0.4;
  scale_mat(2, 2) = 0.4;

  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat = (scale_mat * view_mat).eval().cast<float>();

  mve::OrthoParams ortho;
  ortho.left = static_cast<float>(camera.frustum().left);
  ortho.right = static_cast<float>(camera.frustum().right);
  ortho.bottom = static_cast<float>(camera.frustum().bottom);
  ortho.top = static_cast<float>(camera.frustum().top);
  ortho.height = image->height();
  ortho.width = image->width();

  mve::TriangleMesh::Ptr mesh = mve::geom::depthmap_triangulate(image, nullptr, ortho);

  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat2 = mat.inverse();

  math::Matrix4f ctw(mat2.data());
  mve::geom::mesh_transform(mesh, ctw);
  mesh->recalc_normals(false, true); // Remove this?



  mve::TriangleMesh::VertexList const &mverts(mesh->get_vertices());
  mve::TriangleMesh::ConfidenceList &mconfs(mesh->get_vertex_confidences());
  mve::TriangleMesh::ValueList &vvalues(mesh->get_vertex_values());

  mesh->ensure_normals();

  // PoissonScaleNormals(mconfs, &mesh->get_vertex_normals());

  // Per-vertex confidence down-weighting boundaries.
  mve::geom::depthmap_mesh_confidences(mesh, 4);

  std::vector<float> mvscale;
  mvscale.resize(mverts.size(), 0.0f);
  mve::MeshInfo mesh_info(mesh);
  for (std::size_t p = 0; p < mesh_info.size(); ++p) {
    mve::MeshInfo::VertexInfo const &vinf = mesh_info[p];
    for (std::size_t k = 0; k < vinf.verts.size(); ++k) {
      mvscale[p] += (mverts[p] - mverts[vinf.verts[k]]).norm();
    }
    mvscale[p] /= static_cast<float>(vinf.verts.size());
    mvscale[p] *= FLAGS_scale_factor;
  }
  vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());

  return mesh;
}

#if 0

void PrepareScene(const string &tensor_dir) {
  using boost::filesystem::path;

  const string full_tensor_dir = FileIO::FullOutPath(tensor_dir);

  auto read_tensors = [full_tensor_dir](const vector<string> &subdir_names) {
    // Get list of all files to be read.
    map<string, vector<string>> subdir_filenames;
    // tensor_dir must have a directory named "index".
    subdir_filenames["index"] = FileIO::RegularFilesInDirectory((path(full_tensor_dir) / "index").string());
    size_t num_files = subdir_filenames["index"].size();
    for (const auto &subdir_name : subdir_names) {
      Expects(subdir_name != "index");
      // Sorted filenames.
      auto filenames = FileIO::RegularFilesInDirectory((path(full_tensor_dir) / subdir_name).string());
      Ensures(filenames.size() > 0);
      Ensures(num_files == filenames.size());
      subdir_filenames[subdir_name] = filenames;

      for (int i = 0; i < num_files; ++i) {
        const auto a = fs::basename(filenames[i]);
        const auto b = fs::basename(subdir_filenames["index"][i]);
        Ensures(a == b);
      }
    }

    for (int i = 0; i < num_files; ++i) {
      vector<int> index_shape;
      vector<int> index;
      FileIO::ReadTensorData<int>(subdir_filenames["index"][i], &index_shape, &index);
      Ensures(index_shape.size() == 1);
      const int batch_size = index_shape[0];
      Ensures(batch_size > 0);

      for (const auto &item : subdir_filenames) {
        vector<int> shape;
        vector<float> float_data;
        FileIO::ReadTensorData<float>(item.second[i], &shape, &float_data);
        Ensures(shape.size() == 5);
        Ensures(shape[0] == batch_size);
        Ensures(shape[1] == 6);  // number of views
        Ensures(shape[2] > 0);  // height
        Ensures(shape[3] > 0);  // width
        Ensures(shape[4] == 1);  // channel

        const int num_views = shape[1];
        const int num_values_per_image = shape[2] * shape[3] * shape[4];
        for (int b = 0; b < batch_size; ++b) {
          map<string, shared_ptr<mve::Image<float>>> view_images;

          for (int v = 0; v < num_views; ++v) {
            const int offset = num_values_per_image * (b * num_views + v);
            auto start = float_data.begin() + offset;
            auto end = start + num_values_per_image;
            mve::FloatImage::Ptr image = mve::FloatImage::create(shape[3], shape[2], shape[4]);
            Ensures(num_values_per_image == image->get_value_amount());
            std::copy(start, end, image->get_data_pointer());
            view_images[item.first] = image;
          }

          // emit view_images
        }

      }

    }

  };

  auto target_depth_filenames = read_tensors("placeholder_target_depth");
  auto index_filenames = read_tensors("index");

  LOG(INFO) << "Reading " << index_filenames[0];
  vector<int> index_shape;
  vector<int> index;
  FileIO::ReadTensorData<int>(index_filenames[0], &index_shape, &index);

  vector<int> target_depth_shape;
  vector<float> target_depth;
  FileIO::ReadTensorData<float>(target_depth_filenames[0], &target_depth_shape, &target_depth);

  Ensures(target_depth_shape[0] == index.size());
  Ensures(target_depth_shape[1] == 6);

  mv::Examples test_examples;
  Data::LoadExamples(FileIO::FullOutPath("splits/shrec12_examples_opo/test.bin"), &test_examples);

  auto frustum = FrustumParams();
  frustum.near = Data::kNear;
  frustum.far = Data::kFar;
  Vec3 eye{0, 0, 1};
  Vec3 lookat{0, 0, 0};
  Vec3 up{0, 1, 0};
  auto camera = OrthographicCamera(eye, lookat, up, frustum);

  vector<unique_ptr<Camera>> cams;
  Data::SixViews(camera, &cams);

  for (int i = 0; i < index.size(); ++i) {
    int ind = index[i];

    vector<int> shape{target_depth_shape[2], target_depth_shape[3], target_depth_shape[4]};
    int num_elements = shape[0] * shape[1] * shape[2];

    mve::FloatImage::Ptr image = mve::FloatImage::create(
        target_depth_shape[1], target_depth_shape[0], target_depth_shape[2]);

    std::copy(target_depth.data(), target_depth.data() + image->get_value_amount(),
              image->get_data_pointer()
    );

    for (int j = 0; j < 6; ++j) {
      int id = j;
      mve::View::Ptr view = mve::View::create();
      view->set_id(id);

      std::string mve_fname = "view_" + util::string::get_filled(id, 4) + ".mve";

      FileIO::PrepareDir("/tmp/0001/views");
      auto view_path = path("/tmp/0001/views") / mve_fname;

      auto view_mat = cams[j]->view_mat();
      Mat33 rotation = view_mat.topLeftCorner(3, 3);
      Vec3 translation = view_mat.topRightCorner(3, 1);

      Eigen::Matrix<float, 4, 4, Eigen::RowMajor> mat(view_mat.cast<float>());
      mve::CameraInfo cam;
      cam.set_transformation(mat.data());
      cam.ppoint[0] = 0;
      cam.ppoint[1] = 0;
      cam.flen = -1;
      cam.paspect = -1;
      view->set_camera(cam);
      view->save_view_as(view_path.string());

      mve::OrthoParams ortho;
      ortho.left = -1;
      ortho.right = 1;
      ortho.bottom = -1;
      ortho.top = 1;
      ortho.height = target_depth_shape[2];
      ortho.width = target_depth_shape[3];

      mve::TriangleMesh::Ptr mesh = mve::geom::depthmap_triangulate(image, nullptr, cam, ortho);
      mve::TriangleMesh::VertexList const &mverts(mesh->get_vertices());
      mve::TriangleMesh::ConfidenceList &mconfs(mesh->get_vertex_confidences());
      mve::TriangleMesh::ValueList &vvalues(mesh->get_vertex_values());

      mesh->ensure_normals();

      // PoissonScaleNormals(mconfs, &mesh->get_vertex_normals());

      // Per-vertex confidence down-weighting boundaries.
      mve::geom::depthmap_mesh_confidences(mesh, 4);

      std::vector<float> mvscale;
      mvscale.resize(mverts.size(), 0.0f);
      mve::MeshInfo mesh_info(mesh);
      for (std::size_t p = 0; p < mesh_info.size(); ++p) {
        mve::MeshInfo::VertexInfo const &vinf = mesh_info[p];
        for (std::size_t k = 0; k < vinf.verts.size(); ++k) {
          mvscale[p] += (mverts[p] - mverts[vinf.verts[k]]).norm();
        }
        mvscale[p] /= static_cast<float>(vinf.verts.size());
        mvscale[p] *= FLAGS_scale_factor;
      }
      vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());
    }

  }

}

#endif

}
}
