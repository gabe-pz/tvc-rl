#include "dataLog.h"

void returnToCsv(int batchAmount, int episode, float averageTotalReturn){
    if(episode == batchAmount){
        std::ofstream file("data/returnData.csv");
        file << "episode,averageTotalReturn" << "\n";
        file << episode << "," << averageTotalReturn << "\n";
        file.close();
    }
    else{
        std::ofstream file("data/returnData.csv", std::ios::app);
        file << episode << "," << averageTotalReturn << "\n";
        file.close();
    }
}
