#pragma once

#include <array>

std::array<float, 400> gradientLogPolicy_1(const std::array<std::array<float, 4>, 100>& w1, const std::array<float, 100>& preActVector, const std::array<float, 100> actVector, const std::array<float, 2>& terms_i);

