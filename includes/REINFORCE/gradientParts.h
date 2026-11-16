#pragma once

#include <array>

//******DERIVATIVE FUNCTIONS******
template<std::size_t N>
float derivativeActWrtPreAct(const std::array<float, N>& preActVector, int n);

template<std::size_t rows, std::size_t cols>//rows and cols of the weight matrix for current layer
float derivativePreActWrtAct(const std::array<std::array<float, cols>, rows>& wMat, int n, int m);

template<typename T, std::size_t N>
float derivativePreActWrtWeight(const std::array<T, N>& inputVector, int j);


//******GRADIENT PARTS******
std::array<float, 400> gradientLogPolicy_1(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1,
        const std::array<float, 100>& y2, const std::array<double, 4>& stateVec, const std::array<float, 2>& terms_i, int meanIndex);
std::array<float, 100> gradientLogPolicy_2(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1,
        const std::array<float, 100>& y2, const std::array<float, 2>& terms_i, int meanIndex);


std::array<float, 10000> gradientLogPolicy_3(const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y2, const std::array<float, 100>& a1, 
        const std::array<float, 2>& terms_i, int meanIndex);
std::array<float, 100> gradientLogPolicy_4(const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y2, const std::array<float, 2>& terms_i, int meanIndex);


std::array<float, 400> gradientLogPolicy_5(const std::array<float, 100>& a2, const std::array<float, 2>& terms_i, int meanIndex);
std::array<float, 4> gradientLogPolicy_6(const std::array<float, 2>& terms_i, int meanIndex);
