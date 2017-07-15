//
// Created by daeyun on 6/20/17.
//

#include "string_utils.h"

#include <iomanip>
#include <sstream>
#include <boost/algorithm/string.hpp>

namespace mvshape {

std::string WithLeadingZeros(int value, int num_digits) {
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(num_digits) << value;
  return stream.str();
}

std::string ToLower(const std::string &s) {
  std::string ret = s;
  boost::algorithm::to_lower(ret);
  return ret;
}

}
