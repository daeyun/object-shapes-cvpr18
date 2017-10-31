//
// Created by daeyun on 6/28/17.
//

#include "transforms.h"

#include <limits>
#include <gsl/gsl_assert>

#include "glog/logging.h"

namespace mvshape {

constexpr float kBackgroundValue = std::numeric_limits<float>::quiet_NaN();

double ReplaceZerosWithBackgroundValue(cv::Mat *image) {
  double sum = 0;
  int count = 0;
  for (int i = 0; i < image->rows; i++) {
    for (int j = 0; j < image->cols; j++) {
      float depth_value = image->at<float>(i, j);
      if (depth_value == 0) {
        image->at<float>(i, j) = kBackgroundValue;
      } else if (std::isfinite(depth_value)) {
        sum += depth_value;
        ++count;
      } else {
        throw std::runtime_error("Non-finite floating point value found in image.");
      }
    }
  }
  return sum / count;
}

void RescaleAndRecenter(const cv::Mat &image, tuple<int, int> height_width, int padding, cv::Mat *out) {
  // TODO: Only square images are supported for now.
  // TODO: Only depth images are supported for now.
  Expects(image.cols == image.rows);
  Expects(get<0>(height_width) == get<1>(height_width));

  double in_out_ratio = static_cast<double>(image.cols) / get<0>(height_width);

  bool is_image_empty = true;
  int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float depth_value = image.at<float>(i, j);
      if (depth_value > 0) {
        min_x = std::min(min_x, j);
        min_y = std::min(min_y, i);
        max_x = std::max(max_x, j);
        max_y = std::max(max_y, i);
        is_image_empty = false;
      }
    }
  }

  if (is_image_empty) {
    LOG(WARNING) << "Empty image found.";
  }

  int h = max_y - min_y + 1;
  int w = max_x - min_x + 1;

  auto center = tuple<int, int> {min_y + h * 0.5, min_x + w * 0.5};

  int ystart = get<0>(center) - h / 2;
  int xstart = get<1>(center) - w / 2;

  cv::Rect roi(xstart, ystart, w, h);

  double h_ratio = static_cast<double>(h) / get<0>(height_width);
  double w_ratio = static_cast<double>(w) / get<1>(height_width);

  double im_scale;
  if (h_ratio > w_ratio) {
    im_scale = static_cast<double>(get<0>(height_width) - padding * 2) / h + 1e-8;
  } else {
    im_scale = static_cast<double>(get<1>(height_width) - padding * 2) / w + 1e-8;
  }

  cv::Mat roi_image = image(roi);
  cv::Mat resized_roi_image;
  cv::resize(roi_image, resized_roi_image,
             {static_cast<int>(w * im_scale + 0.5),
              static_cast<int>(h * im_scale + 0.5)}, 0, 0, cv::INTER_NEAREST);

  double value_scale = im_scale * in_out_ratio;
  resized_roi_image = resized_roi_image * value_scale;

  *out = cv::Mat(get<0>(height_width), get<1>(height_width), resized_roi_image.type(), cv::Scalar(0.0));
  int h_start = static_cast<int>((get<0>(height_width) - resized_roi_image.rows) / 2 + 0.5);
  int w_start = static_cast<int>((get<1>(height_width) - resized_roi_image.cols) / 2 + 0.5);

  roi.x = w_start;
  roi.y = h_start;
  roi.width = resized_roi_image.cols;
  roi.height = resized_roi_image.rows;

  Ensures(roi.y + roi.height <= out->cols);
  Ensures(roi.x + roi.width <= out->rows);

  resized_roi_image.copyTo((*out)(roi));

  auto mean = ReplaceZerosWithBackgroundValue(out);

  for (int i = 0; i < out->rows; i++) {
    for (int j = 0; j < out->cols; j++) {
      float depth_value = out->at<float>(i, j);
      if (std::isfinite(depth_value)) {
        out->at<float>(i, j) -= mean;
      }
    }
  }

  Ensures(out->rows == get<0>(height_width));
  Ensures(out->cols == get<1>(height_width));
}

}
