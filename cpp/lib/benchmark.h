//
// Created by daeyun on 6/14/17.
//
#pragma once

#include <ratio>
#include <iomanip>

namespace mvshape {

constexpr double kMicro = std::micro::num / static_cast<double>(std::micro::den);

long MicroSecondsSinceEpoch();

template<typename Unit=std::ratio<1, 1>>
double TimeSinceEpoch() {
  constexpr double ratio = kMicro * Unit::den / Unit::num;
  return MicroSecondsSinceEpoch() * ratio;
}

class Timer {
 public:
  explicit Timer(const std::string &name);

  void Tic();

  void Toc();

  template<typename Unit=std::ratio<1, 1>>
  double OperationsPer() {
    constexpr double ratio = Unit::num / (kMicro * Unit::den);
    return running_average_microseconds_ * ratio;
  }

  template<typename Unit=std::ratio<1, 1>>
  double Elapsed() {
    constexpr double ratio = kMicro * Unit::den / Unit::num;
    return running_average_microseconds_ * ratio;
  }

  static constexpr double kEmaDecay = 0.98;

 private:
  double running_average_microseconds_ = -1;
  long last_seen_;
  const std::string name_;
};

}
