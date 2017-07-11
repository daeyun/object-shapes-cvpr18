//
// Created by daeyun on 6/28/17.
//
#pragma once

#include <opencv2/opencv.hpp>

#include "common.h"

namespace mvshape {

// Sets NaN where pixel value is 0. All values in `image` must be finite.
// Returns the mean of non-zero values.
double ReplaceZerosWithBackgroundValue(cv::Mat *image);

void RescaleAndRecenter(const cv::Mat &image, tuple<int, int> height_width, int padding, cv::Mat *out);

}
