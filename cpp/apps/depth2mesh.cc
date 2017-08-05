#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <fstream>

#include "third_party/repos/mve/libs/math/octree_tools.h"
#include "third_party/repos/mve/libs/util/system.h"
#include "third_party/repos/mve/libs/util/arguments.h"
#include "third_party/repos/mve/libs/util/tokenizer.h"
#include "third_party/repos/mve/libs/util/file_system.h"
#include "third_party/repos/mve/libs/util/ini_parser.h"
#include "third_party/repos/mve/libs/mve/depthmap.h"
#include "third_party/repos/mve/libs/mve/mesh_info.h"
#include "third_party/repos/mve/libs/mve/mesh_io.h"
#include "third_party/repos/mve/libs/mve/mesh_io_ply.h"
#include "third_party/repos/mve/libs/mve/mesh_tools.h"
#include "third_party/repos/mve/libs/mve/scene.h"
#include "third_party/repos/mve/libs/mve/view.h"

#define META_FILE "meta.ini"

struct AppSettings {
  std::string scenedir;
  std::string outmesh;
  std::string dmname = "depth-L0";
  std::string image = "undistorted";
  bool with_normals = true;
  bool with_scale = true;
  bool with_conf = true;
  bool poisson_normals = false;
  bool is_ply_binary = true;
  float min_valid_fraction = 0.0f;
  float scale_factor = 2.5f; /* "Radius" of MVS patch (usually 5x5). */
  std::vector<int> ids;
};

void poisson_scale_normals(mve::TriangleMesh::ConfidenceList const &confs,
                           mve::TriangleMesh::NormalList *normals) {
  if (confs.size() != normals->size())
    throw std::invalid_argument("Invalid confidences or normals");
  for (std::size_t i = 0; i < confs.size(); ++i)
    normals->at(i) *= confs[i];
}

int main(int argc, char **argv) {
  util::system::register_segfault_handler();
  util::system::print_build_timestamp("MVE Scene to Mesh");

  /* Setup argument parser. */
  util::Arguments args;
  args.set_exit_on_error(true);
  args.set_nonopt_minnum(2);
  args.set_nonopt_maxnum(2);
  args.set_helptext_indent(25);
  args.set_usage("Usage: depth2mesh [ OPTS ] SCENE_DIR MESH_OUT");
  args.add_option('S', "scale-factor", true, "Factor for computing scale values [2.5]");
  args.add_option('B', "is_ply_binary", true, "Whether the output ply file is binary or ascii.");
  args.parse(argc, argv);

  /* Init default settings. */
  AppSettings conf;
  conf.scenedir = args.get_nth_nonopt(0);
  conf.outmesh = args.get_nth_nonopt(1);

  /* Scan arguments. */
  while (util::ArgResult const *arg = args.next_result()) {
    if (arg->opt == nullptr)
      continue;

    switch (arg->opt->sopt) {
      case 'S': conf.scale_factor = arg->get_arg<float>();
        break;
      case 'B': conf.is_ply_binary = arg->get_arg<bool>();
        break;
      default: throw std::runtime_error("Unknown option");
    }
  }

  if (util::string::right(conf.outmesh, 4) != ".ply") {
    std::cerr << "Error: Invalid MESH_OUT. Must end with .ply" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (conf.poisson_normals) {
    conf.with_normals = true;
    conf.with_conf = true;
  }

  std::cout << "Using depthmap \"" << conf.dmname
            << "\" and color image \"" << conf.image << "\"" << std::endl;

  /* Load scene. */
  mve::Scene::Ptr scene(mve::Scene::create(conf.scenedir));

  /* Iterate over views and get points. */
  mve::Scene::ViewList &views(scene->get_views());
#pragma omp parallel for schedule(dynamic)
  for (std::size_t i = 0; i < views.size(); ++i) {
    mve::View::Ptr view = views[i];
    if (view == nullptr)
      continue;

    std::size_t view_id = static_cast<size_t>(view->get_id());
    if (!conf.ids.empty() && std::find(conf.ids.begin(),
                                       conf.ids.end(), view_id) == conf.ids.end())
      continue;

    mve::FloatImage::Ptr dm = view->get_float_image(conf.dmname);
    if (dm == nullptr)
      continue;

    if (conf.min_valid_fraction > 0.0f) {
      float num_total = static_cast<float>(dm->get_value_amount());
      float num_recon = 0;
      for (int j = 0; j < dm->get_value_amount(); ++j)
        if (dm->at(j) > 0.0f)
          num_recon += 1.0f;
      float fraction = num_recon / num_total;
      if (fraction < conf.min_valid_fraction) {
        std::cout << "View " << view->get_name() << ": Fill status "
                  << util::string::get_fixed(fraction * 100.0f, 2)
                  << "%, skipping." << std::endl;
        continue;
      }
    }

    mve::ByteImage::Ptr ci;
    if (!conf.image.empty())
      ci = view->get_byte_image(conf.image);

#pragma omp critical
    {
      std::cout << "Processing view \"" << view->get_name()
                << "\"" << (ci != nullptr ? " (with colors)" : "")
                << "..." << std::endl;
    }

    /* Triangulate depth map. */
    mve::CameraInfo const &cam = view->get_camera();

    std::string const fname = util::fs::join_path(view->get_directory(), META_FILE);
    /* Open file and read key/value pairs. */
    std::ifstream in(fname.c_str());
    if (!in.good())
      throw util::FileException(fname, "Error opening");
    std::map<std::string, std::string> meta_ini;
    util::parse_ini(in, &meta_ini);

    mve::OrthoParams ortho;
    ortho.top = util::string::convert<float>(meta_ini["orthographic_camera.top"]);
    ortho.left = util::string::convert<float>(meta_ini["orthographic_camera.left"]);
    ortho.bottom = util::string::convert<float>(meta_ini["orthographic_camera.bottom"]);
    ortho.right = util::string::convert<float>(meta_ini["orthographic_camera.right"]);
    ortho.height = dm->height();
    ortho.width = dm->width();

    mve::TriangleMesh::Ptr mesh;
    mesh = mve::geom::depthmap_triangulate(dm, ci, cam, ortho);
    mve::TriangleMesh::VertexList const &mverts(mesh->get_vertices());
    mve::TriangleMesh::ConfidenceList &mconfs(mesh->get_vertex_confidences());
    mve::TriangleMesh::ValueList &vvalues(mesh->get_vertex_values());

    if (conf.with_normals)
      mesh->ensure_normals();

    /* If confidence is requested, compute it. */
    if (conf.with_conf) {
      /* Per-vertex confidence down-weighting boundaries. */
      mve::geom::depthmap_mesh_confidences(mesh, 4);
    }

    if (conf.poisson_normals)
      poisson_scale_normals(mconfs, &mesh->get_vertex_normals());

    /* If scale is requested, compute it. */
    std::vector<float> mvscale;
    if (conf.with_scale) {
      mvscale.resize(mverts.size(), 0.0f);
      mve::MeshInfo mesh_info(mesh);
      for (std::size_t j = 0; j < mesh_info.size(); ++j) {
        mve::MeshInfo::VertexInfo const &vinf = mesh_info[j];
        for (std::size_t k = 0; k < vinf.verts.size(); ++k)
          mvscale[j] += (mverts[j] - mverts[vinf.verts[k]]).norm();
        mvscale[j] /= static_cast<float>(vinf.verts.size());
        mvscale[j] *= conf.scale_factor;
      }
    }

    vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());

    mve::geom::SavePLYOptions opts;
    opts.write_vertex_normals = true;
    opts.write_vertex_values = true;
    opts.write_vertex_confidences = true;
    opts.format_binary = conf.is_ply_binary;
    std::ostringstream ply_name;

#pragma omp critical
    {
      ply_name << util::string::left(conf.outmesh, conf.outmesh.length() - 4) << "_" << std::setw(4)
               << std::setfill('0') << i << ".ply";
      std::cout << "Writing " << ply_name.str() << std::endl;
      mve::geom::save_ply_mesh(mesh, ply_name.str(), opts);
    }

    dm.reset();
    ci.reset();
    view->cache_cleanup();
  }

  return EXIT_SUCCESS;
}
