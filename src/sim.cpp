#include <array>
#include <random>

#include "../includes/graphics/raylibFunctions.h"
#include "../includes/physics/rocketPhysics.h"
#include "../includes/readWrite/rwParameters.h"

#include "../includes/control/slewServos.h"
#include "../includes/mlp/mlpFunctions.h"

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
    int n = (int)(simTime*GEN_FREQ) + 2;
    double U = 5.0;//average wind velocity
    double intensity = 0.20;//turbulence intensity 
    double sigmaU = intensity * U;
    
    //wind turbulence buffers
    std::vector<double> pinkU = generatePinkNoise(n, seed); std::vector<double> pinkV = generatePinkNoise(n, seed + 1); std::vector<double> pinkW = generatePinkNoise(n, seed + 2);
    std::array<std::vector<double>, 3> pinkNoise = {pinkU, pinkV, pinkW};

    

    //*******STATE ARRAY INITALIZATIONS*****
    std::array<double, 4> stateQ = {1.0, 0.0, 0.0, 0.0};

    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> position = {0.0, 0.0, 0.0};
    std::array<double, 3> lastPosition = {0.0, 0.0, 0.0};

    std::array<double, 3> angularAcceleration = {0.0, 0.0, 0.0};
    std::array<double, 3> angularVelocity = {0.0, 0.0, 0.0};
    std::array<double, 2> angularPosition ={0.0, 0.0};



    //*****RAYLIB INITALIZATION*****
    int FPS = 60;  
    Camera3D camera = raylibInit(FPS, position);
    int iterationsPerFrame = (1/(dt*FPS));//raylib sync 



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
    
    //******PARAMETERS****** 
    //weight matrices 
    std::array<std::array<float, 4>, 100> w1;
    std::array<std::array<float, 100>, 100> w2; 
    std::array<std::array<float, 100>, 4> w3; 
    
    //bias vectors
    std::array<float, 100> b1, b2; 
    std::array<float, 4> b3;

    loadParameters(w1, w2, w3, b1, b2, b3);//load parameters

    //*****SIM***** 
    while (!WindowShouldClose()){
        if(!landed){
            lastPosition = position;

            //main physics loop
            for(int i = 0; i < iterationsPerFrame; i++){

                //*****CONTROL TIME*****
                t += dt;
                timeSinceLastControl += dt; 



                //******CONTROL*****
                if(timeSinceLastControl >= controlDt){
                    timeSinceLastControl = 0.0;//reset control time

                    std::array<double, 4> stateVector = {angularPosition[0], angularPosition[1], angularVelocity[0], angularVelocity[1]};

                    //propagate state vector through MLP
                    std::array<float, 2> mus = mlpOutputControl(stateVector, w1, w2, w3, b1, b2, b3);

                    //raw actions
                    float rawActionX = mus[0];
                    float rawActionY = mus[1];

                    actionX = 0.087*tanh(rawActionX); 
                    actionY = 0.087*tanh(rawActionY);
                }



                //*****SERVO SLEW******
                //keep movements of gimbal within the physical limits
                slewServo(currentServoX, actionX, dt);
                slewServo(currentServoY, actionY, dt);
               


                //******PHYSICS UPDATE******
                stateUpdate(dt, t, U, sigmaU, currentServoX, currentServoY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularAcceleration, angularVelocity, angularPosition);
                


                //*****LANDING CHECK*****
                if(position[2] < 0 && t > 0.5){
                    landed = true; 
                    break;
                }
                
                
            }
        }

        raylibDrawRocket(distanceToThrustVector, centerOfGravity, stateQ, position, lastPosition, camera);

    }


    CloseWindow();    
    
    return 0; 
}

