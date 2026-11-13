#pragma once


#include <array>
#include <raylib.h> 
#include <rlgl.h>  

#include "../math/rocketMath.h"


Vector3 toRaylib(const std::array<double, 3>& p);
Camera3D raylibInit(int FPS, const std::array<double, 3>& startPosition);
void raylibDrawRocket(const double distanceToThrustVector, const double centerOfGravity, const std::array<double, 4>& stateQ, const std::array<double, 3> position, 
        const std::array<double, 3> lastPos, Camera3D& camera);
