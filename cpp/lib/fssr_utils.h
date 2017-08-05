//
// Created by daeyun on 8/4/17.
//

#pragma once

#include <string>
#include <vector>
#include <map>

#include <third_party/repos/mve/libs/mve/image.h>
#include <third_party/repos/mve/libs/mve/mesh.h>

#include "cpp/lib/batch_queue.h"
#include "camera.h"

namespace mvshape {
namespace fssr {

struct FloatImages {
  std::map<std::string, std::vector<std::shared_ptr<mve::Image<float>>>> images;
  int index = 0;
};

template<typename T>
void WriteMvei(const std::string &filename, const std::vector<int> &shape, const std::vector<T> &data);

template<typename T>
void WriteMvei(const std::string &filename, const std::vector<int> &shape, const void *data, size_t size_bytes);

// e.g. ReadSavedTensors("tf_out/0040_000011400/NOVELCLASS", {"out_depth"})
std::thread ReadSavedTensors(const std::string &tensor_dir, const std::vector<std::string> &subdir_names,
                             mvshape::concurrency::BatchQueue<FloatImages> *queue);


mve::TriangleMesh::Ptr Depth2MeshSingle(const mve::Image<float>::Ptr& image, const mvshape::Camera& camera);

void Depth2Mesh(const string &tensor_dir, const string &out_dir);

void PrepareScene(const std::string &tensor_dir);

}
}
