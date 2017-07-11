//
// Created by daeyun on 7/3/17.
//

#pragma once

#include <random>
#include <algorithm>
#include <gsl/gsl_assert>

namespace mvshape {
namespace Random {

std::mt19937 &Engine() {
  thread_local static std::mt19937 engine{std::random_device{}()};
  return engine;
}

template<class Iter>
void Shuffle(Iter begin, Iter end) {
  std::shuffle(begin, end, Engine());
}

int UniformInt(int n) {
  std::uniform_int_distribution<decltype(n)> dist{0, n - 1};
  return dist(Engine());
}

// If n is greater than pool size, some items will be chosen more than once.
template<class T>
void ChooseN(size_t n, std::vector<T> *mutable_pool, std::vector<T> *choices) {
  Expects(!mutable_pool->empty());
  auto left_size = mutable_pool->size();
  for (int i = 0; i < n; ++i) {
    if (left_size <= 0) {
      left_size = mutable_pool->size();
    }
    auto& chosen = mutable_pool->at(UniformInt(left_size));
    choices->push_back(chosen);
    std::swap(chosen, mutable_pool->at(left_size - 1));
    --left_size;
  }
}

}
}

