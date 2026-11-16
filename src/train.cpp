#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>

#include "../includes/mlp/mlpFunctions.h"
#include "../includes/physics/rocketPhysics.h"
#include "../includes/control/slewServos.h"
#include "../includes/REINFORCE/gradientTerm.h"

int main(void){
    //*****RANDOM NUM*****
    std::random_device rd;//random unsigned int 
    std::mt19937 gen(rd());//mersenne twister generator, which gives psudo-random number when call gen


    //*****SIMULATION SETTINGS*****
    //main settings
    const double dt = 0.0001;
    const double simTime = tBurn; 
    double t = 0.0;

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt;//such that start loop with control



    //*******STATE ARRAY INITALIZATIONS***** 
    std::array<double, 6> stateVector;

    std::array<double, 4> stateQ;

    std::array<double, 3> velocity, position, angularVelocity, angularAccleration;

    std::array<double, 2> angularPosition;



    //*****CONTROL*****
    //actions init 
    float actionX = 0.0f;
    float actionY = 0.0f;
   
    //set servos offset
    std::uniform_real_distribution<float> dist(-0.017f, 0.034f);//misalignment between -1 and 2 deg
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



    //*****SEQUENCES AND MATRICES*****  
    //dynamic matrices for pre-act and act vectors
    std::vector<std::array<float, 100>> y1Mat;
    std::vector<std::array<float, 100>> y2Mat;
    std::vector<std::array<float, 4>> y3Mat;

    std::vector<std::array<float, 100>> a1Mat;
    std::vector<std::array<float, 100>> a2Mat;
    std::vector<std::array<float, 4>> a3Mat;

    //SAR, matrices and sequence
    std::vector<std::array<double, 6>> stateVectorMat;
    std::vector<std::array<float, 2>> actionsMat;
    std::vector<float> rewardSeq;

    //return
    float returnT = 0.0f;

    //*****TRAINING PARAMETERS*****
    int numEpisodes = 10000;
    int numIterations = 0;
    bool epTerminated = false;


    //*****HYPERPARAMETERS******
    float alpha = 0.000001;//step size
    float gamma = 0.97;

    int a = 190;//reward angular exp constant
    int b = 8;//reward angularV constant
    int c = -10;//termination reward
                
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
        epTerminated = false;
        t = 0.0; 
        timeSinceLastControl = controlDt; 
        numIterations = 0;

        //vectors
        stateVector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        stateQ = {1.0, 0.0, 0.0, 0.0};

        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};
        
        angularAccleration = {0.0, 0.0, 0.0};
        angularVelocity = {0.0, 0.0, 0.0};
        angularPosition = {0.0, 0.0};
        
        //control 
        currentServoX = 0.0f;
        currentServoY = 0.0f;

        //matrices 
        y1Mat.clear();
        y2Mat.clear();
        y3Mat.clear();

        a1Mat.clear();
        a2Mat.clear();
        a3Mat.clear();

        //matries and seq
        stateVectorMat.clear();
        actionsMat.clear();
        rewardSeq.clear();


        //*****EPISODE GENERATION******
        for(int i = 0; i < (int)(simTime/dt); i++){
            //time
            t += dt;
            timeSinceLastControl += dt;

            //*****CONTROL*****
            if(timeSinceLastControl >= controlDt){
                timeSinceLastControl = 0.0;

                //*****REWARD*****
                //ensure only getting reward after initial action
                if(i > 0){
                    //analyze state for reward
                    if(std::abs(angularPosition[0]) > 0.17 || std::abs(angularPosition[1]) > 0.17){

                        //reward for case where angles greater than termination angles
                        rewardSeq.push_back(c);
                        epTerminated = true;

                        break;//terminate episode
                    }

                    //reward for otherwise 
                    float expRewardTerm = std::exp(-a*(angularPosition[0]*angularPosition[0] + angularPosition[1]*angularPosition[1]));
                    float squaredSumRewardTerm = b*(angularVelocity[0]*angularVelocity[0] + angularVelocity[1]*angularVelocity[1]);
                    float reward = expRewardTerm - squaredSumRewardTerm;
                    
                    rewardSeq.push_back(reward);
                    
                }

                //*****STATE*****
                stateVector = {angularPosition[0], angularPosition[1], angularVelocity[0], angularVelocity[1], angularAccleration[0], angularAccleration[1]}; 
                stateVectorMat.push_back(stateVector);

                //*****MLP PROPAGATION*****
                mlpOutputTraining mlpOut = mlpOutTrain(stateVector, w1, w2, w3, b1, b2, b3);

                //log outputs for use in training 
                y1Mat.push_back(mlpOut.y_1);
                a1Mat.push_back(mlpOut.a_1);

                y2Mat.push_back(mlpOut.y_2);
                a2Mat.push_back(mlpOut.a_2);

                y3Mat.push_back(mlpOut.y_out);

                //*****ACTION*****
                float meanX = mlpOut.y_out[0];
                float logSigmaX = mlpOut.y_out[1];
                float standardDevX_raw = std::exp(logSigmaX);//e^log(sigma_x) = sigma_x
                float standardDevX = std::clamp(standardDevX_raw, 0.25f, 5.0f);
                
                
                float meanY = mlpOut.y_out[2];
                float logSigmaY = mlpOut.y_out[3];
                float standardDevY_raw = std::exp(logSigmaY);//e^log(sigma_y) = sigma_y                                       
                float standardDevY = std::clamp(standardDevY_raw, 0.25f, 5.0f);


                std::normal_distribution<float> gaussianX(meanX, standardDevX);
                std::normal_distribution<float> gaussianY(meanY, standardDevY);

                float rawActionX = gaussianX(gen);
                float rawActionY = gaussianY(gen);

                //tanh squashing
                actionX = 0.087*std::tanh(rawActionX);//multiply by 0.087 rads(5 degs) such that scale up to be within action space
                actionY = 0.087*std::tanh(rawActionY);

                actionsMat.push_back({rawActionX, rawActionY});//log the actions

                //increment next step for use of indexing in update rule 
                numIterations ++;
            }
            
            //*****UPDATE IN SIM*****
            //slew the servos to ensure only moving within physical limits
            slewServo(currentServoX, actionX, dt);//x slew
            slewServo(currentServoY, actionY, dt);//y slew
            
            //physics update
            stateUpdate(dt, t, U, sigmaU, currentServoX, currentServoY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularAccleration, angularVelocity, angularPosition);
        }

        //log final reward at end of episode if not terminated 
        if(!epTerminated){

            //reward for otherwise 
            float expRewardTerm = std::exp(-a*(angularPosition[0]*angularPosition[0] + angularPosition[1]*angularPosition[1]));
            float squaredSumRewardTerm = b*(angularVelocity[0]*angularVelocity[0] + angularVelocity[1]*angularVelocity[1]);
            float reward = expRewardTerm - squaredSumRewardTerm;
            
            rewardSeq.push_back(reward);
        }

        
        float totalEpisodeReturn = 0.0f;
        std::array<float, 11204> accumlatedGradTerm = {0};
        bool isNan = false;

        for(int t = 0; t < numIterations; t++){
            //*****RETURN_t*****
            returnT = 0.0f;

            //compute return at step t 
            for(int k = t; k < numIterations; k++){
                returnT += std::pow(gamma, k-t)*rewardSeq.at(k);

            }
            if(t == 0) totalEpisodeReturn = returnT;

            //****GRADIENT TERM*****
            //gradLP_x
            float meanX = y3Mat[t][0];
            float logSigmaX = y3Mat[t][1];

            float standardDevX_raw = std::exp(logSigmaX);
            float standardDevX = std::clamp(standardDevX_raw, 0.25f, 5.0f);

            float actionX = actionsMat[t][0];
            std::array<float, 11204> gradLogPolicy_x = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionX, meanX, standardDevX, 0);


            //gradLP_y
            float meanY = y3Mat[t][2];
            float logSigmaY = y3Mat[t][3];

            float standardDevY_raw = std::exp(logSigmaY);
            float standardDevY = std::clamp(standardDevY_raw, 0.25f, 5.0f);

            float actionY = actionsMat[t][1];
            
            std::array<float, 11204> gradLogPolicy_y = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionY, meanY, standardDevY, 2);

            //assemble full gradient term 
            std::array<float, 11204> gradTerm = gradientTerm(gradLogPolicy_x, gradLogPolicy_y, returnT, alpha, gamma, t);

            //accumalte gradientTerm 
            for(int i = 0; i < 11204; i++){
                isNan = std::isnan(gradTerm[i]);
                if(isNan){
                    std::cout << "gradTerm[" << i << "] = " << gradTerm[i] << " at time step " << t << std::endl;
                    break;
                }
                accumlatedGradTerm[i] += gradTerm[i];
            }
        }
        if(isNan) break;

        std::cout << "Total return for episode " << e << " was G = " << totalEpisodeReturn << std::endl;

        //******UPDATE PARAMTERS*****
        updateParameters(accumlatedGradTerm, w1, w2, w3, b1, b2, b3);
    }

    return 0;
}
