//
// Created by daeyun on 6/20/17.
//
#pragma once

#include <algorithm>

#include <gsl/gsl_assert>
#include <Eigen/Dense>

#include "common.h"

namespace mvshape {

template<typename T>
vector<T> SplitValues(const string &str, char delimiter = ',') {
  std::vector<T> values;
  std::stringstream stream(str);
  T value;
  while (true) {
    stream >> value;
    if (stream.fail()) {
      throw std::runtime_error("Invalid values: " + str);
    } else {
      values.push_back(value);
      if (stream.peek() == delimiter) {
        stream.ignore();
      }
    }

    if (stream.eof()) {
      break;
    }
  }
  return values;
}

template<int rows, int cols = 1>
Eigen::Matrix<double, rows, cols> ParseMatrix(const string &str, char delimiter = ',') {
  auto values = SplitValues<double>(str, delimiter);
  Ensures(values.size() == rows * cols);
  return Eigen::Matrix<double, rows, cols>(values.data());
}

std::string WithLeadingZeros(int value, int num_digits);

std::string ToLower(const std::string &s);

}
