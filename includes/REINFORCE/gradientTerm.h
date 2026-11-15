#pragma once

#include <array> 
#include "gradientParts.h"

std::array<float, 12204> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVector, float action_i, float mean_i, float standardDev_i, int meanIndex);

void assembleGragLogPolicy_i();
