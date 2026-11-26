#pragma once

#include <array> 
#include <cmath> 

std::array<float, 2> reward(const std::array<double, 3>& angularVelocity, const std::array<double, 2>& angularPosition, float a, float b, float c, float d, 
        float rTerm, double angularThX, double angularThY);
