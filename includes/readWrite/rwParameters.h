#pragma once 

#include <array> 
#include <fstream> 
#include <iostream> 

void writeParameters(const std::array<std::array<float, 4>, 100>& w1, const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& b1, 
        const std::array<float, 100>& b2, const std::array<float, 4>& b3);

void loadParameters(std::array<std::array<float, 4>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3);
