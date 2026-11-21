#include "logStats.h"


void logStats(int currentEpisode, int epBatchAmount, int& timeStepSum, float& totalReturnSum, double& timeSum, const std::vector<std::array<double, 4>>& stateVecMat, 
        const std::vector<std::array<float, 4>>& y3Mat){
    double averageFlightTime = timeSum / (epBatchAmount*1.0);
    float averageReturn = totalReturnSum / (epBatchAmount*1.0f);
    int averageTimeStep = timeStepSum / epBatchAmount; 


    //average angularPosition 
    double angularPositionXSum = 0.0;
    double angularPositionYSum = 0.0;
    for(int i = 0; i < stateVecMat.size(); i++){
        angularPositionXSum += stateVecMat[i][0];
        angularPositionYSum += stateVecMat[i][1];
    }
    double averageAngularPositionX = angularPositionXSum / (epBatchAmount*1.0);
    double averageAngularPositionY = angularPositionYSum / (epBatchAmount*1.0);

    //average logSigmas
    float logSigmaXSum = 0.0f;
    float logSigmaYSum = 0.0f;
    for(int i = 0; i < y3Mat.size(); i++){
        logSigmaXSum += y3Mat[i][1];
        logSigmaYSum += y3Mat[i][3];
    }
    float averageLogSigmaX = logSigmaXSum / (epBatchAmount*1.0);
    float averageLogSigmaY = logSigmaYSum / (epBatchAmount*1.0);
    
    //prints
    std::cout << "**********AVERAGES FOR EPISODE " << currentEpisode << "**********" << std::endl;
    std::cout << "TOTAL RETURN(G_0): " << averageReturn << std::endl;
    std::cout << "NUMBER OF TIME STEPS: " << averageTimeStep << std::endl;
    std::cout << "FLIGHT TIME: " << averageFlightTime << std::endl;
    std::cout << "LOG_SIGMA_X: " << averageLogSigmaX << std::endl;
    std::cout << "LOG_SIGMA_Y: " << averageLogSigmaY << std::endl;
    std::cout << "ANGULAR_POS_X: " << rad2deg(averageAngularPositionX) << "(degs)" << std::endl;
    std::cout << "ANGULAR_POS_Y: " << rad2deg(averageAngularPositionY) << "(degs)" << std::endl;
    std::cout << "\n" << std::endl;
    
    //log to csv
    returnToCsv(epBatchAmount, currentEpisode, averageReturn);

    //reset
    totalReturnSum = 0.0f;
    timeStepSum = 0;
    timeSum = 0.0;
}




