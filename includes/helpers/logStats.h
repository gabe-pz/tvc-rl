#pragma once 

#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include "../math/rocketMath.h"
#include "../readWrite/dataLog.h"

void logStats(int currentEpisode, int epBatchAmount, int& timeStepSum, float& totalReturnSum, double& timeSum, const std::vector<std::array<double, 4>>& stateVecMat, 
        const std::vector<std::array<float, 4>>& y3Mat);
