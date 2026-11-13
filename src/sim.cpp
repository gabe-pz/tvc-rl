#include <random>


int main(void){
    //*****RANDOM NUM*****
    std::random_device rd; 
    std::mt19937 gen(rd());


    //*****SIMULATION SETTINGS*****
    //main settings
    const double dt = 0.0001;
    const int simTime = 15;
    double t = 0.0;

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt; 


    //*******STATE ARRAY INITALIZATIONS*****
    std::array<double, 4> stateQ = {1.0, 0.0, 0.0, 0.0};

    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> position = {0.0, 0.0, 0.0};

    std::array<double, 3> angularVelocity = {0.0, 0.0, 0.0};
    std::array<double, 2> psi ={0.0, 0.0};


    //*****SERVOS***** 
    //servos offset
    std::uniform_real_distribution<float> dist(0.017f, 0.034f);//misalignment between 1 and 2 deg
    float servosXOffset = dist(gen);
    float servosYOffset = dist(gen);

    //servos
    float currentServoX = 0.0f;
    float currentServoY = 0.0f;


    //*****CONTROL*****
    //action init 
    float actionX = 0.0f;
    float actionY = 0.0f;



    return 0; 
}

