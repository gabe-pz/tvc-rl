#include "reward.h"




std::array<float, 2> reward(const std::array<double, 4>& stateVector, float a, float b, float c, float d, double angularThX, double angularThY){

    double angularPositionX = stateVector[0];
    double angularPositionY = stateVector[1];
     
    //******CHECK TERMINATION CASE******
    if(std::abs(angularPositionX) > angularThX || std::abs(angularPositionY) > angularThY){
        return {d, 1.0f};
    }

    //*****REWARD FOR OTHERWISE*****
    float expRewardTerm = a*std::exp(-b*(angularPositionX*angularPositionX + angularPositionY*angularPositionY));
    float squaredSumRewardTerm = c*(angularPositionX*angularPositionX+angularPositionY*angularPositionY);
    float reward = expRewardTerm - squaredSumRewardTerm;

    return {reward, 0.0f};
}
