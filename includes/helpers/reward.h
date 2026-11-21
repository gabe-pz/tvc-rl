#pragma once

#include <array> 
#include <cmath> 

std::array<float, 2> reward(const std::array<double, 4>& stateVector, float a, float b, float c, float d, float f, double angularThX, double angularThY);
