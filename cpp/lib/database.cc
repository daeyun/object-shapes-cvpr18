//
// Created by daeyun on 6/28/17.
//

#include "database.h"

#include <glog/logging.h>

#include <gsl/gsl_assert>
#include <sqlite_modern_cpp.h>
#include <boost/filesystem.hpp>
#include <unordered_map>
#include <boost/algorithm/string.hpp>

#include "common.h"
#include "flags.h"
#include "file_io.h"
#include "proto/dataset.pb.h"
#include "third_party/repos/nlohmann_json/src/json.hpp"

namespace mvshape {
namespace Data {

namespace fs = boost::filesystem;

Vec3 ToVec3(const vector<float> &values) {
  Expects(3 == values.size());
  return {values[0], values[1], values[2]};
}

// Check if `view_name` exists.
void CheckView(sqlite::database &db, const std::string &view_name) {
  LOG(INFO) << "Checking whether view " << view_name << " exists";
  std::stringstream stream;
  stream << "SELECT count(*) FROM sqlite_master WHERE type='view' AND name='" << view_name << "'";
  string exists_query = stream.str();
  bool exists;
  db << exists_query >> exists;

  if (!exists) {
    LOG(ERROR) << "View " << view_name << " does not exist.";
    throw std::runtime_error("view " + view_name + " does not exist in the database.");
  }
  LOG(INFO) << "View " << view_name << " exists";
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
    Expects(!dataset_name.empty());

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
  LOG(INFO) << "Reading renderables from " << sqlite_file;
  sqlite::sqlite_config config;
  config.flags = sqlite::OpenFlags::READONLY;
  sqlite::database db(FileIO::FullDataPath(sqlite_file), config);
  auto ret = ReadRenderables(db, view_name, out);
  LOG(INFO) << ret << " examples to render";
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

int ReadExamples(sqlite::database &db, const std::string &view_name, json *examples) {
  CheckView(db, view_name);

  std::stringstream stream;

  stream << R"(SELECT * FROM `)" << view_name << "`;";
  string query = stream.str();
  LOG(INFO) << "Running query " << query;
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

  // Find renderings associated with each example.
  stream.str("");
  stream << R"(SELECT t0.example_id, t0.rendering_id FROM `exampleobjectrendering` AS `t0`;)";
  query = stream.str();
  LOG(INFO) << "Running query " << query;

  std::unordered_map<int, vector<int>> example_id_to_rendering_ids;
  db << query >> [&](int example_id,
                     int rendering_id) {
    auto it = example_id_to_split.find(example_id);
    if (it != example_id_to_split.end()) {
      example_id_to_rendering_ids[example_id].push_back(rendering_id);
    }
  };
  // Find tags associated with each example.
  stream.str("");
  stream << R"(SELECT t0.example_id, tag.name FROM `exampletag` AS `t0` LEFT JOIN tag ON t0.tag_id = tag.id;)";
  query = stream.str();
  LOG(INFO) << "Running query " << query;
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

  // Convert renderings in database to protobuf.
  vector<mv::Rendering> renderings;
  std::map<int, mv::Rendering> renderings_by_id;
  // view "all_renderings" must exist.
  ReadRenderables(db, "all_renderings", &renderings);
  for (const auto &rendering : renderings) {
    auto rendering_id = rendering.id();
    renderings_by_id[rendering_id] = rendering;
  }

//  std::map<mv::Split, vector<mv::Example>> split_examples;
  LOG(INFO) << "Processing retrieved examples.";

  // alias
  json& all_examples = *examples;

  // For each example id from the database, make an Example protobuf which consists of multiple renderings.
  int example_count = 0;
  for (const auto &kv : example_id_to_split) {
    int example_id = kv.first;
    mv::Split split = kv.second;

#if 0
    for (int rendering_id : example_id_to_rendering_ids[example_id]) {
      Expects(renderings_by_id.find(rendering_id) != renderings_by_id.end());
      mv::Rendering rendering = renderings_by_id[rendering_id];
      if (rendering.rendering_type() == mv::Rendering_Type_DEPTH) {
        if (rendering.set_size() == 1) {
          // As a special case, there are two types of "single depth".
          if (rendering.is_normalized()) {
            Expects(!example.has_single_depth());
            *example.mutable_single_depth() = rendering;
          } else {
            // Same image before normalizing transform. Used to derive the observed point cloud.
            // Could be higher resolution than `single_depth`.
            Expects(!example.has_single_depth_raw());
            *example.mutable_single_depth_raw() = rendering;
          }
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
      } else {
        throw std::runtime_error("Unknown rendering type.");
      }


    }

    for (auto tag : example_id_to_tags[example_id]) {
      example.add_tags(tag);
    }

    // Insert to all examples proto.
    mv::Split split_save = split;
    bool exists = examples->find(split_save) != examples->end();
    // A hack to split a large training set into two.
    if (exists && split_save == mv::TRAIN && example_id_to_split.size() > 1200000
        && (*examples)[split_save].examples_size() > example_id_to_split.size() / 2) {
      split_save = mv::TRAIN2;
    }

    if (!exists) {
      auto &examples_split = (*examples)[split_save]; // insert
      (*examples)[split_save].set_dataset_name(dataset);
      (*examples)[split_save].set_split_name(split_save);
    }
    *(*examples)[split_save].add_examples() = example;

    // TODO: add new tags to examples.

    example_count++;
    LOG_EVERY_N(INFO, 50000) << "processed " << example_count << " examples";
#endif

    json example;

    for (int rendering_id : example_id_to_rendering_ids[example_id]) {
      Expects(renderings_by_id.find(rendering_id) != renderings_by_id.end());
      mv::Rendering rendering = renderings_by_id[rendering_id];

      if (rendering.rendering_type() == mv::Rendering_Type_DEPTH) {
        if (rendering.set_size() == 1) {
          // As a special case, there are two types of "single depth".
          if (rendering.is_normalized()) {
            example["input_depth"]["filename"] = rendering.filename();
            example["input_depth"]["id"] = rendering.id();

            // example metadata
            example["mesh_filename"] = rendering.mesh_filename();
            example["input_camera"]["eye"] = { rendering.eye(0), rendering.eye(1), rendering.eye(2) };
            example["input_camera"]["up"] = { rendering.up(0), rendering.up(1), rendering.up(2) };
            example["input_camera"]["lookat"] = { rendering.lookat(0), rendering.lookat(1), rendering.lookat(2) };
            example["input_camera"]["fov"] = rendering.fov();
            example["input_camera"]["scale"] = rendering.scale();
            example["input_camera"]["id"] = rendering.camera_id();
            example["category"]["name"] = rendering.category_name();
            example["category"]["id"] = rendering.category_id();

            // dataset metadata. overwritten multiple times.
            all_examples["input_resolution"] = rendering.resolution();
          } else {
            // Skip
          }
        } else {
          example["target_depth"]["filename"] = rendering.filename();
          example["target_depth"]["id"] = rendering.id();

          // example metadata
          example["target_camera"]["eye"] = { rendering.eye(0), rendering.eye(1), rendering.eye(2) };
          example["target_camera"]["up"] = { rendering.up(0), rendering.up(1), rendering.up(2) };
          example["target_camera"]["lookat"] = { rendering.lookat(0), rendering.lookat(1), rendering.lookat(2) };
          example["target_camera"]["fov"] = rendering.fov();
          example["target_camera"]["scale"] = rendering.scale();
          example["target_camera"]["id"] = rendering.camera_id();

          // dataset metadata. overwritten multiple times.
          all_examples["target_resolution"] = rendering.resolution();
        }
      } else if (rendering.rendering_type() == mv::Rendering_Type_NORMAL) {
        // skip
      } else if (rendering.rendering_type() == mv::Rendering_Type_RGB) {
        if (rendering.set_size() == 1) {
          example["input_rgb"]["filename"] = rendering.filename();
          example["input_rgb"]["id"] = rendering.id();
        } else {
          // skip
        }
      } else if (rendering.rendering_type() == mv::Rendering_Type_VOXELS) {
        example["target_voxels"]["filename"] = rendering.filename();
        example["target_voxels"]["id"] = rendering.id();
      } else {
        throw std::runtime_error("Unknown rendering type.");
      }
    }


    all_examples[mv::Split_Name(split)].push_back(example);

    example_count++;
    LOG_EVERY_N(INFO, 50000) << "processed " << example_count << " examples";
  }

  return count;
}

int ReadExamples(const string &sqlite_file,
                 const std::string &view_name,
                 json *examples) {
  LOG(INFO) << "Reading examples from view " << view_name << " in " << sqlite_file;
  sqlite::sqlite_config config;
  config.flags = sqlite::OpenFlags::READONLY;
  sqlite::database db(FileIO::FullDataPath(sqlite_file), config);
  return ReadExamples(db, view_name, examples);
}

}
}
