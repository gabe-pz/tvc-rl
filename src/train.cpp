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
    std::array<double, 4> stateVector;

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
    std::array<std::array<float, 4>, 100> w1;
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
    std::vector<std::array<double, 4>> stateVectorMat;
    std::vector<std::array<float, 2>> actionsMat;
    std::vector<float> rewardSeq;

    //return
    float returnT = 0.0f;

    //*****TRACKING VALUES*****
    int numIterations = 0;
    bool epTerminated = false;
    double timeSum = 0.0;


    //*****HYPERPARAMETERS****** 
    int numEpisodes = 50*1000;

    float alpha = 0.001;//step size
    float gamma = 0.92;
    float clippingVal = 5.5f;

    float lowerBoundLogSigma = -3.0f;
    float upperBoundLogSigma = 0.2f;

    float a = 290.5f;//reward angular exp constant
    float b = 0.5f;//reward angularV constant
    float c = 0.0f;//termination reward
                
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
        stateVector = {0.0, 0.0, 0.0, 0.0};

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
                    if(std::abs(angularPosition[0]) > 0.27 || std::abs(angularPosition[1]) > 0.27){

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
                stateVector = {angularPosition[0], angularPosition[1], angularVelocity[0], angularVelocity[1]}; 
                stateVectorMat.push_back(stateVector);

                //*****MLP PROPAGATION*****
                mlpOutputTraining mlpOut = mlpOutTrain(stateVector, w1, w2, w3, b1, b2, b3);

                //log outputs for use in training 
                y1Mat.push_back(mlpOut.y_1);
                a1Mat.push_back(mlpOut.a_1);

                y2Mat.push_back(mlpOut.y_2);
                a2Mat.push_back(mlpOut.a_2);

                y3Mat.push_back(mlpOut.y_out);

                //*****NETWORK OUTPUTS******
                //x outputs
                float meanX = mlpOut.y_out[0];
                float logSigmaXRaw = mlpOut.y_out[1];
                float logSigmaX = std::clamp(logSigmaXRaw, lowerBoundLogSigma, upperBoundLogSigma);


                float standardDevX = std::exp(logSigmaX);//e^log(sigma_x) = sigma_x
               
                    
                //y outputs 
                float meanY = mlpOut.y_out[2];
                float logSigmaYRaw = mlpOut.y_out[3];
                float logSigmaY = std::clamp(logSigmaYRaw, lowerBoundLogSigma, upperBoundLogSigma);


                float standardDevY = std::exp(logSigmaY);//e^log(sigma_y) = sigma_y                                       


                //*******ACTIONS******
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
            
            //*****UPDATE IN ENV*****
            //slew the servos to ensure only moving within physical limits
            slewServo(currentServoX, actionX, dt);//x slew
            slewServo(currentServoY, actionY, dt);//y slew
            
            //physics update
            stateUpdate(dt, t, U, sigmaU, currentServoX, currentServoY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularAccleration, angularVelocity, angularPosition);
        }
        timeSum += t;

        //log final reward at end of episode if not terminated 
        if(!epTerminated){

            //reward for otherwise 
            float expRewardTerm = std::exp(-a*(angularPosition[0]*angularPosition[0] + angularPosition[1]*angularPosition[1]));
            float squaredSumRewardTerm = b*(angularVelocity[0]*angularVelocity[0] + angularVelocity[1]*angularVelocity[1]);
            float reward = expRewardTerm - squaredSumRewardTerm;
            
            rewardSeq.push_back(reward);
        }

        
        float totalEpisodeReturn = 0.0f;
        std::array<float, 11004> accumlatedGrad = {0};
        bool isNan = false;
        bool logSigmaXClamped = false;
        bool logSigmaYClamped = false;

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
            float logSigmaXRaw = y3Mat[t][1];
            float logSigmaX = std::clamp(logSigmaXRaw, lowerBoundLogSigma, upperBoundLogSigma);

            if(logSigmaX == lowerBoundLogSigma || logSigmaX == upperBoundLogSigma) logSigmaXClamped = true;
            else logSigmaXClamped = false;

            float standardDevX = std::exp(logSigmaX);

            float actionX = actionsMat[t][0];
            std::array<float, 11004> gradLogPolicy_x = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionX, meanX, standardDevX, 0, logSigmaXClamped);


            //gradLP_y
            float meanY = y3Mat[t][2];
            float logSigmaYRaw = y3Mat[t][3];
            float logSigmaY = std::clamp(logSigmaYRaw, lowerBoundLogSigma, upperBoundLogSigma);

            if(logSigmaY == lowerBoundLogSigma || logSigmaY == upperBoundLogSigma) logSigmaYClamped = true;
            else logSigmaYClamped = false;

            float standardDevY = std::exp(logSigmaY);

            float actionY = actionsMat[t][1];
            
            std::array<float, 11004> gradLogPolicy_y = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionY, meanY, standardDevY, 2, logSigmaYClamped);

            //assemble full gradient term 
            std::array<float, 11004> gradTerm_t = gradientTerm_t(gradLogPolicy_x, gradLogPolicy_y, returnT, t);

            //accumalte gradientTerm 
            for(int i = 0; i < 11004; i++){
                isNan = std::isnan(gradTerm_t[i]);
                if(isNan){
                    std::cout << "gradTerm_t[" << i << "] = " << gradTerm_t[i] << " at time step " << t << std::endl;
                    break;
                }

                accumlatedGrad[i] += gradTerm_t[i];
            }

            if(isNan) break;
        }
        if(isNan) break;

        //******APPLY UPDATE TO PARAMETERS******
        std::array<float, 11004> gradTerm = gradientTerm(accumlatedGrad, alpha, clippingVal);


        updateParameters(gradTerm, w1, w2, w3, b1, b2, b3);


        if((e % 1000 == 0 && e > 0) || e == numEpisodes-1){
            std::cout << "************************************************************" << std::endl;
            std::cout << "Total discounted return for episode " << e << " was G = " << totalEpisodeReturn << std::endl;
            std::cout << "Average flight time was: " << timeSum /(e*1.0) << "s" << std::endl;
        }
    }

    return 0;
}
