#pragma once

#include <array> 
#include <cmath>
#include "gradientParts.h"

std::array<float, 12204> assembleGragLogPolicy(const std::array<float, 600>& gLP1, const std::array<float, 100>& gLP2, const std::array<float, 10000>& gLP3, 
        const std::array<float, 100>& gLP4, const std::array<float, 400>& gLP5, const std::array<float, 4>& gLP6);

std::array<float, 12204> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVector, float action_i, float mean_i, float standardDev_i, int meanIndex);


std::array<float, 12204> gradientTerm(const std::array<float, 12204>& gradLPx, const std::array<float, 12204>& gradLPy, float returnT, float alpha, float gamma, int t);

void updateParameters(const std::array<float, 12204>& accumaltedGradTerm, std::array<std::array<float, 6>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, 
        std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, std::array<float, 100>& b2, std::array<float, 4>& b3);
