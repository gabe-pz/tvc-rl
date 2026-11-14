#pragma once

#include <array>

template<std::size_t N>
float derivativeActWrtPreAct(const std::array<float, N>& preActVector, int n);



template<std::size_t rows, std::size_t cols>//rows and cols of the weight matrix for current layer
float derivativePreActWrtAct(const std::array<std::array<float, cols>, rows>& wMat, int n, int m);

template<typename T, std::size_t N>
float derivativeActWrtWeight(const std::array<T, N>& inputVector, int j);
