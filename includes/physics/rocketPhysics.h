#pragma once

#include <array> 
#include <cmath>
#include <random>

#include "windGen.h" 
#include "rocketProperties.h" 
#include "../math/rocketMath.h"


inline constexpr double centerOfGravity = 0.405;
inline constexpr double distanceToThrustVector = 0.6477;




void stateUpdate(const double dt, double t, double U, double sigmaU, float actionX, float actionY, float servoXOffset, float servoYOffset, 
        std::array<std::vector<double>, 3> pinkNoise, std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 4>& stateQ, 
        std::array<double, 4>& stateQMeasure, std::array<double, 3>& angularAccleration, std::array<double, 3>& angularVelocity, std::array<double, 2>& psi, 
        std::array<double, 3>& angularVelocityMeasure, std::array<double, 2>& psiMeasure, std::mt19937& gen);
