#include "cpp/lib/common.h"

#pragma once

namespace mvshape {

// `data` should be allocated by the caller.
template<typename T>
uint32_t ReadSingleBatch(const vector<string> &filenames, const vector<int> &shape, T *data);

}
