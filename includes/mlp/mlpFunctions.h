#pragma once 

#include <array>
#include <random>

//*****PARAMETERS INIT*****
void initParameters(std::array<std::array<float, 6>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3);

//*****ACTIVATION FUNCTION*****
float ReLU(float y);

//*****PRE-ACTIVATION AND ACTIVATION*****
template<typename T, std::size_t rows, std::size_t cols>//rows and cols of the weight matrix for current layer
std::array<float, rows> preActivationVec(const std::array<std::array<float, cols>, rows>& weightMatrix, const std::array<T, cols>& vec, const std::array<float, rows>& biasVec);

template<std::size_t N>//size of the pre-activation vector 
std::array<float, N> activationsVec(const std::array<float, N>& y_N);
