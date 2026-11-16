#pragma once

#include <array> 
#include <cmath>
#include "gradientParts.h"

std::array<float, 11004> assembleGragLogPolicy(const std::array<float, 400>& gLP1, const std::array<float, 100>& gLP2, const std::array<float, 10000>& gLP3, 
        const std::array<float, 100>& gLP4, const std::array<float, 400>& gLP5, const std::array<float, 4>& gLP6);

std::array<float, 11004> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<float, 100>& a1, const std::array<float, 100>& a2, const std::array<double, 4>& stateVector,
        float action_i, float mean_i, float standardDev_i, int meanIndex, bool logSigmaClamped);


std::array<float, 11004> gradientTerm_t(const std::array<float, 11004>& gradLPx, const std::array<float, 11004>& gradLPy, float returnT, int t);

std::array<float, 11004> gradientTerm(const std::array<float, 11004>& accumaltedGradTerm, float alpha, float clipVal);

void updateParameters(const std::array<float, 11004>& gradTerm, std::array<std::array<float, 4>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, 
        std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, std::array<float, 100>& b2, std::array<float, 4>& b3);
