//
// Created by daeyun on 6/28/17.
//

#include "database.h"

#include <glog/logging.h>

#include <gsl/gsl_assert>
#include <sqlite_modern_cpp/sqlite_modern_cpp.h>
#include <boost/filesystem.hpp>
#include <unordered_map>
#include <boost/algorithm/string.hpp>

#include "common.h"
#include "flags.h"
#include "file_io.h"
#include "proto/dataset.pb.h"

namespace mvshape {
namespace Data {

namespace fs = boost::filesystem;

Vec3 ToVec3(const vector<float> &values) {
  Expects(3 == values.size());
  return {values[0], values[1], values[2]};
}

void CheckView(sqlite::database &db, const std::string &view_name) {
  std::stringstream stream;
  stream << "SELECT count(*) FROM sqlite_master WHERE type='view' AND name='" << view_name << "'";
  string exists_query = stream.str();
  bool exists;
  db << exists_query >> exists;

  if (!exists) {
    LOG(ERROR) << "View " << view_name << " does not exist.";
    throw std::runtime_error("view " + view_name + " does not exist in the database.");
  }
}

int ReadRenderables(sqlite::database &db, const std::string &view_name, vector<mvshape_dataset::Rendering> *out) {
  CheckView(db, view_name);
  std::stringstream stream;
  stream << R"(SELECT * FROM `)" << view_name << "`;";
  string query = stream.str();
  int count = 0;

  db << query >> [&](int rendering_id,
                     string target_filename,
                     int resolution,
                     int num_channels,
                     int set_size,
                     bool is_normalized,
                     string rendering_type,
                     string mesh_filename,
                     int object_id,
                     vector<float> eye,
                     vector<float> up,
                     vector<float> lookat,
                     float fov,
                     float scale,
                     int camera_id,
                     string category_name,
                     int category_id,
                     string dataset_name) {
    Expects(eye.size() == 3);
    Expects(up.size() == 3);
    Expects(lookat.size() == 3);

    mvshape_dataset::Rendering rendering;
    rendering.set_id(rendering_id);
    rendering.set_filename(target_filename);
    rendering.set_resolution(resolution);
    rendering.set_num_channels(num_channels);
    rendering.set_set_size(set_size);
    rendering.set_is_normalized(is_normalized);

    if (rendering_type == "depth") {
      rendering.set_rendering_type(mvshape_dataset::Rendering_Type_DEPTH);
    } else if (rendering_type == "normal") {
      rendering.set_rendering_type(mvshape_dataset::Rendering_Type_NORMAL);
    } else if (rendering_type == "rgb") {
      rendering.set_rendering_type(mvshape_dataset::Rendering_Type_RGB);
    } else if (rendering_type == "voxels") {
      rendering.set_rendering_type(mvshape_dataset::Rendering_Type_VOXELS);
    }

    rendering.set_mesh_filename(mesh_filename);
    rendering.set_object_id(object_id);
    for (int i = 0; i < 3; ++i) {
      rendering.add_eye(eye[i]);
      rendering.add_up(up[i]);
      rendering.add_lookat(lookat[i]);
    }
    rendering.set_fov(fov);
    rendering.set_scale(scale);
    rendering.set_camera_id(camera_id);
    rendering.set_category_name(category_name);
    rendering.set_category_id(category_id);

    out->push_back(rendering);
    count++;
  };

  return count;
}

int ReadRenderables(const string &sqlite_file, const std::string &view_name, vector<mvshape_dataset::Rendering> *out) {
  sqlite::sqlite_config config;
  config.flags = sqlite::OpenFlags::READONLY;
  sqlite::database db(FileIO::FullDataPath(sqlite_file), config);
  auto ret = ReadRenderables(db, view_name, out);
  return ret;
}

string ToString(const mvshape_dataset::Rendering &rendering) {
  std::ostringstream stream;

  stream << mvshape_dataset::Rendering_Type_Name(rendering.rendering_type()) << ", "
         << rendering.set_size() << ", "
         << rendering.resolution() << ", "
         << rendering.num_channels() << ", "
         << "cam " << rendering.camera_id() << ", "
         << rendering.mesh_filename() << ", " << rendering.filename();
  return stream.str();
}

int ReadExamples(sqlite::database &db, const std::string &view_name, std::map<mv::Split, mv::Examples> *examples) {
  CheckView(db, view_name);

  std::stringstream stream;

  stream << R"(SELECT * FROM `)" << view_name << "`;";
  string query = stream.str();
  int count = 0;

  std::unordered_map<int, mv::Split> example_id_to_split;
  mv::DatasetName dataset;
  bool dataset_name_init = false;

  db << query >> [&](int example_id,
                     string split_name,
                     string dataset_name) {
    Expects(!split_name.empty());
    mv::Split split = static_cast<mv::Split>(mv::Split_descriptor()->FindValueByName(
        boost::algorithm::to_upper_copy(split_name))->number());
    example_id_to_split[example_id] = split;
    count++;

    Expects(!dataset_name.empty());
    mv::DatasetName ds = static_cast<mv::DatasetName >(mv::DatasetName_descriptor()->FindValueByName(
        boost::algorithm::to_upper_copy(dataset_name))->number());
    if (!dataset_name_init) {
      dataset_name_init = true;
      dataset = ds;
    }
    Expects(ds == dataset);
  };

  // ---
  stream.str("");
  stream << R"(SELECT t0.example_id, t0.rendering_id FROM `exampleobjectrendering` AS `t0`;)";
  query = stream.str();
  std::unordered_map<int, vector<int>> example_id_to_rendering_ids;
  db << query >> [&](int example_id,
                     int rendering_id) {
    auto it = example_id_to_split.find(example_id);
    if (it != example_id_to_split.end()) {
      example_id_to_rendering_ids[example_id].push_back(rendering_id);
    }
  };
  stream.str("");
  stream << R"(SELECT t0.example_id, tag.name FROM `exampletag` AS `t0` LEFT JOIN tag ON t0.tag_id = tag.id;)";
  query = stream.str();
  std::unordered_map<int, vector<mv::Tag>> example_id_to_tags;
  db << query >> [&](int example_id,
                     string tag_name) {
    auto it = example_id_to_split.find(example_id);
    if (it != example_id_to_split.end()) {
      auto tag = static_cast<mv::Tag>(mv::Tag_descriptor()->FindValueByName(
          boost::algorithm::to_upper_copy(tag_name))->number());
      example_id_to_tags[example_id].push_back(tag);
    }
  };

  vector<mv::Rendering> renderings;
  std::map<int, mv::Rendering> renderings_by_id;
  // view "all_renderings" must exist.
  ReadRenderables(db, "all_renderings", &renderings);
  for (const auto &rendering : renderings) {
    renderings_by_id[rendering.id()] = rendering;
  }

  std::map<mv::Split, vector<mv::Example>> split_examples;

  for (const auto &kv : example_id_to_split) {
    int example_id = kv.first;
    mv::Split split = kv.second;
    mv::Example example;
    for (int rendering_id : example_id_to_rendering_ids[example_id]) {
      mv::Rendering rendering = renderings_by_id[rendering_id];
      if (rendering.rendering_type() == mv::Rendering_Type_DEPTH) {
        if (rendering.set_size() == 1) {
          Expects(!example.has_single_depth());
          *example.mutable_single_depth() = rendering;
        } else {
          Expects(!example.has_multiview_depth());
          *example.mutable_multiview_depth() = rendering;
        }
      } else if (rendering.rendering_type() == mv::Rendering_Type_NORMAL) {
        if (rendering.set_size() == 1) {
          throw std::runtime_error("not implemented");
        } else {
          Expects(!example.has_multiview_normal());
          *example.mutable_multiview_normal() = rendering;
        }
      } else if (rendering.rendering_type() == mv::Rendering_Type_RGB) {
        if (rendering.set_size() == 1) {
          Expects(!example.has_single_rgb());
          *example.mutable_single_rgb() = rendering;
        } else {
          Expects(!example.has_multiview_rgb());
          *example.mutable_multiview_rgb() = rendering;
        }
      } else if (rendering.rendering_type() == mv::Rendering_Type_VOXELS) {
        Expects(!example.has_voxels());
        *example.mutable_voxels() = rendering;
      }
    }
    for (auto tag : example_id_to_tags[example_id]) {
      example.add_tags(tag);
    }
    split_examples[split].push_back(example);
  }

  for (const auto &kv: split_examples) {
    mv::Split split = kv.first;
    (*examples)[split].set_dataset_name(dataset);
    (*examples)[split].set_split_name(split);
    for (const auto &example : kv.second) {
      *(*examples)[split].add_examples() = example;
    }
    // TODO: tags
  }

  return count;
}

int ReadExamples(const string &sqlite_file,
                 const std::string &view_name,
                 std::map<mv::Split, mv::Examples> *examples) {
  LOG(INFO) << "Reading examples from view " << view_name << " in " << sqlite_file;
  sqlite::sqlite_config config;
  config.flags = sqlite::OpenFlags::READONLY;
  sqlite::database db(FileIO::FullDataPath(sqlite_file), config);
  return ReadExamples(db, view_name, examples);
}

}
}
