#include "reward.h"




std::array<float, 2> reward(const std::array<double, 4>& stateVector, float a, float b, float c, float d, float rTerm, double angularThX, double angularThY){

    double angularPositionX = stateVector[0];
    double angularPositionY = stateVector[1]; 

    double angularVelocityX = stateVector[2];
    double angularVelocityY = stateVector[3];

    //******CHECK TERMINATION CASE******
    if(std::abs(angularPositionX) > angularThX || std::abs(angularPositionY) > angularThY){
        return {rTerm, 1.0f};
    }

    //*****REWARD FOR OTHERWISE*****
    float r1 = a*std::exp(-b*(angularPositionX*angularPositionX + angularPositionY*angularPositionY));
    float r2 = -c*(angularPositionX*angularPositionX + angularPositionY*angularPositionY);
    float r3 = -std::exp(-d*(angularPositionX*angularPositionX+angularPositionY*angularPositionY))*(angularVelocityX*angularVelocityX+angularVelocityY*angularVelocityY);

    float reward = r1+r2+r3;

    return {reward, 0.0f};
}
