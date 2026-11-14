#include <random>
#include <vector>
#include <iostream>

#include "../includes/mlp/mlpFunctions.h"
#include "../includes/physics/rocketPhysics.h"
#include "../includes/control/slewServos.h"

int main(void){
    //*****RANDOM NUM*****
    std::random_device rd;//random unsigned int 
    std::mt19937 gen(rd());//mersenne twister generator, which gives psudo-random number when calling, gen


    //*****SIMULATION SETTINGS*****
    //main settings
    const double dt = 0.0001;
    const int simTime = tBurn; 
    double t = 0.0;

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt;//such that start loop with control



    //*******STATE ARRAY INITALIZATIONS*****
    std::array<double, 4> stateQ;

    std::array<double, 3> velocity, position, angularVelocity;

    std::array<double, 2> angularPosition;



    //*****CONTROL*****
    //actions init 
    float actionX = 0.0f;
    float actionY = 0.0f;
   
    //set servos offset
    std::uniform_real_distribution<float> dist(0.017f, 0.034f);//misalignment between 1 and 2 deg
    float servosXOffset = dist(gen);
    float servosYOffset = dist(gen);

    //servos init 
    float currentServoX = 0.0f;
    float currentServoY = 0.0f;


    //******PARAMETERS****** 
    //weight matrices 
    std::array<std::array<float, 6>, 100> w1;
    std::array<std::array<float, 100>, 100> w2; 
    std::array<std::array<float, 100>, 4> w3; 
    
    //bias vectors
    std::array<float, 100> b1, b2; 
    std::array<float, 4> b3;

    //init parameters
    initParameters(w1, w2, w3, b1, b2, b3);
   

    //*****TRAINING PARAMETERS*****
    int numEpisodes = 5*1000*10;
    int numIterations = 0;

    //*****TRAINING LOOP*****
    for(int e = 0; e < numEpisodes; e++){

        //******WIND GEN*****  
        //random wind per episode 
        std::uniform_real_distribution<double> distU(2.0, 8.0);//random average wind speed
        double U = distU(gen);
        
        std::uniform_real_distribution<double> distIntensity(0.10, 0.20);//random turb intensity
        double intensity = distIntensity(gen); 
        
        double sigmaU = intensity * U;

        //generate the random wind
        unsigned int episodeSeed = rd();
        int n = (int)(simTime*GEN_FREQ)+2;

        std::vector<double> pinkU = generatePinkNoise(n, episodeSeed); std::vector<double> pinkV = generatePinkNoise(n, episodeSeed+1); std::vector<double> pinkW = generatePinkNoise(n, episodeSeed+2);
        std::array<std::vector<double>, 3> pinkNoise = {pinkU, pinkV, pinkW};

        //*****RESET ALL*****
        //time and counters
        t = 0.0; 
        timeSinceLastControl = controlDt; 
        numIterations = 0;

        //vectors corresponding to state
        stateQ = {1.0, 0.0, 0.0, 0.0};

        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};

        angularVelocity = {0.0, 0.0, 0.0};
        angularPosition = {0.0, 0.0};
        
        //control 
        currentServoX = 0.0f;
        currentServoY = 0.0f;

        //return and sequences
        
        //*****EPISODE GENERATION******
        for(int i = 0; i < (int)(simTime/dt); i++){

        }

    }

    return 0;
}
