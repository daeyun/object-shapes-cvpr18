//
// Created by daeyun on 6/14/17.
//
#pragma once

#include "file_io.h"

namespace mvshape {
namespace Resources {

/**
 * Converts a filename in `resources` directory to a global path.
 * The file must exist.
 * @param filename e.g. "shaders/object.frag"
 * @return A canonical path.
 * @throw boost::filesystem::filesystem_error
 */
std::string ResourcePath(const std::string &filename);

std::string ReadResource(const std::string &filename);

std::string FindResourceDir();

}
}
