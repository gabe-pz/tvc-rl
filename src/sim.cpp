#include <random>

#include "../includes/graphics/raylibFunctions.h"
#include "../includes/physics/rocketPhysics.h"
#include "../includes/control/slewServos.h"

int main(void){
    //*****RANDOM NUM*****
    std::random_device rd; 
    std::mt19937 gen(rd());



    //*****SIMULATION SETTINGS*****
    //main settings
    const double dt = 0.0001;
    const int simTime = 15; 
    double t = 0.0;
    bool landed = false;

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt;//such that start loop with control



    //*****WIND SETTINGS*****
    //wind generation constants
    unsigned int seed = 12345;
    int n = (int)(simTime * GEN_FREQ) + 2;
    double U = 5.0;  //average wind velocity
    double intensity = 0.20; //turbulence intensity 
    double sigmaU = intensity * U;
    
    //wind turbulence buffers
    std::vector<double> pinkU = generatePinkNoise(n, seed); std::vector<double> pinkV = generatePinkNoise(n, seed + 1); std::vector<double> pinkW = generatePinkNoise(n, seed + 2);
    std::array<std::vector<double>, 3> pinkNoise = {pinkU, pinkV, pinkW};

    

    //*****RAYLIB INITALIZATION*****
    int FPS = 60;  
    Camera3D camera = raylibInit(FPS);
    int iterationsPerFrame = (1/(dt*FPS));//raylib sync 



    //*******STATE ARRAY INITALIZATIONS*****
    std::array<double, 4> stateQ = {1.0, 0.0, 0.0, 0.0};

    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> position = {0.0, 0.0, 0.0};

    std::array<double, 3> angularVelocity = {0.0, 0.0, 0.0};
    std::array<double, 2> psi ={0.0, 0.0};


    //*****SERVOS***** 
    //set servos offset
    std::uniform_real_distribution<float> dist(0.017f, 0.034f);//misalignment between 1 and 2 deg
    float servosXOffset = dist(gen);
    float servosYOffset = dist(gen);

    //servos init 
    float currentServoX = 0.0f;
    float currentServoY = 0.0f;


    //*****CONTROL*****
    //actions init 
    float actionX = 0.0f;
    float actionY = 0.0f;


    //*****SIM***** 
    while (!WindowShouldClose()){
        if(!landed){

            //main physics loop
            for(int i = 0; i < iterationsPerFrame; i++){
                //*****CONTROL TIME*****
                t += dt;
                timeSinceLastControl += dt; 


                //******CONTROL*****
                if(timeSinceLastControl >= controlDt){
                    timeSinceLastControl = 0.0;//reset control time

                    actionX = 0; 
                    actionY = 0;
                }

                //*****SERVO SLEW******
                slewServo(currentServoX, actionX, dt);//keep movements of servos within physical limits
                slewServo(currentServoY, actionY, dt);
                
                //******PHYSICS UPDATE******
                stateUpdate(dt, t, U, sigmaU, currentServoX, currentServoY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularVelocity, psi);
                                
                //*****LANDING CHECK*****
                if(position[2] < 0 && t > 0.5){
                    landed = true; 
                    break;
                }
                
                
            }
        }
        raylibDrawRocket(distanceToThrustVector, centerOfGravity, stateQ, position, camera);
    }


    CloseWindow();    
    
    return 0; 
}

