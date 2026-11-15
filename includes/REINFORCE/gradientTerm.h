#pragma once

#include <array>
#include "../REINFORCE/derivatives.h"

std::array<float, 600> gradientLogPolicy_1(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3,
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVec, const std::array<float, 2>& terms_i, int meanIndex);

std::array<float, 100> gradientLogPolicy_2(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3,
        const std::array<float, 100>& y2, const std::array<float, 2>& terms_i, int meanIndex);

