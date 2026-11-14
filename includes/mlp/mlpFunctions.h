#pragma once 

#include <array>
#include <random>

//*****PARAMETERS INIT*****
void initParameters(std::array<std::array<float, 4>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3);

//*****ACTIVATION FUNCTION*****
float ReLU(float y);

//*****Pre-activations and activations*****
std::array<float, 100> preActivation_1(const std::array<std::array<float, 4>, 100>& w1, const std::array<double, 6>& stateVector, const std::array<float, 100>& b1);
