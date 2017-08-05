//
// Created by daeyun on 4/11/17.
//

#pragma once

#ifndef NDEBUG
#define USE_OMP false
#else
#define USE_OMP true
#endif

#include <iostream>
#include <limits>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <map>
#include <memory>

#include <gsl/gsl_assert>
#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;

using Vec = Eigen::VectorXd;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat44 = Eigen::Matrix<double, 4, 4>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat33 = Eigen::Matrix<double, 3, 3>;
using MatrixX = Eigen::MatrixXd;

using Points3d = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Points2d = Eigen::Matrix<double, 2, Eigen::Dynamic>;
using Points2i = Eigen::Matrix<int, 2, Eigen::Dynamic>;

using std::vector;
using std::array;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_unique;
using std::make_shared;
using std::tuple;
using std::pair;
using std::map;
using std::get;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kOneOverTwoPi = M_1_PI * 0.5;
constexpr double kEpsilon = 1e-9;
constexpr double kDeg2Rad = M_PI / 180.0;
