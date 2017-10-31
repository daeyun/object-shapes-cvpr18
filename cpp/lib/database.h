//
// Created by daeyun on 6/28/17.
//

#pragma once

#include <map>
#include <unordered_map>

#include "common.h"

#include "proto/dataset.pb.h"
#include "third_party/repos/nlohmann_json/src/json.hpp"

namespace mvshape {
namespace Data {

namespace mv = mvshape_dataset;
using json = nlohmann::json;

// sqlite_file should be relative to the data directory and start with a slash.
// e.g. view_name = "shrec12,perspective,viewer_centered,6_views:test"
int ReadRenderables(const string &sqlite_file, const string &view_name, vector<mvshape_dataset::Rendering> *out);
int ReadExamples(const string &sqlite_file,
                 const std::string &view_name,
                 json *examples);
string ToString(const mvshape_dataset::Rendering &rendering);

}
}
