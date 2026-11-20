#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>

#include "../includes/mlp/mlpFunctions.h"

#include "../includes/readWrite/rwParameters.h"

#include "../includes/physics/rocketPhysics.h"
#include "../includes/control/slewServos.h"
#include "../includes/REINFORCE/gradientTerm.h"
#include "../includes/helpers/reward.h"

int main(void){

    //*****RANDOM NUM*****
    std::random_device rd;//random unsigned int 
    std::mt19937 gen(rd());//mersenne twister generator, which gives psudo-random number when call gen



    //*****SIMULATION SETTINGS*****
    //time and ep
    const double dt = 0.0001;
    const double simTime = tBurn; 
    const int numEpisodes = 10000;

    //control
    const double controlDt = 0.05; 



    //*******STATE ARRAY INITALIZATIONS***** 
    std::array<double, 4> stateVector;

    std::array<double, 4> stateQ;

    std::array<double, 3> velocity, position, angularVelocity, angularAccleration;

    std::array<double, 2> angularPosition;



    //*****CONTROL*****
    //actions init 
    float actionX = 0.0f;
    float actionY = 0.0f;
   
    //servos init 
    float currentServoX = 0.0f;
    float currentServoY = 0.0f;

    //gimbal offset
    std::uniform_real_distribution<float> distX(-0.034f, 0.034f);//some misalignment between -2 and 2 degs                                                              
    std::uniform_real_distribution<float> distY(-0.034f, 0.034f);
    
    //set the gimbal offsets
    float servosXOffset = distX(gen);//misalignment x
    float servosYOffset = distY(gen);//misalignment y



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



    //*****STD VECOTRS FOR STORING VALUES*****  
    //dynamic matrices for storing pre-act and act vectors
    std::vector<std::array<float, 100>> y1Mat;
    std::vector<std::array<float, 100>> y2Mat;
    std::vector<std::array<float, 4>> y3Mat;

    std::vector<std::array<float, 100>> a1Mat;
    std::vector<std::array<float, 100>> a2Mat;
    std::vector<std::array<float, 4>> a3Mat;

    //dynamic matrices for storing state and action vectors
    std::vector<std::array<double, 4>> stateVectorMat;
    std::vector<std::array<float, 2>> actionsMat;

    //dynamic vector for storing rewards
    std::vector<float> rewardSeq;


    //*****TRACKING VARIABLES***** 
    //time
    double t = 0.0;
    double timeSum = 0.0;
    double timeSinceLastControl = controlDt;//such that start loop with control 
    int timeSteps = 0; 

    //return 
    float returnT = 0.0f;

    //episode state
    bool epTerminated = false;

    

    //******HYPERPARAMETERS******
    float alpha = 0.001;
    float gamma = 0.99;
    float gradientClipValue = 5.5f;
    float lowerBoundLogSigma = -4.0f;
    float upperBoundLogSigma = -1.0f;



    //******REWARD FUNCTION CONSTANTS*****
    //reward_t constants
    float a = 5.0f;//constant that scales exp term 
    float b = 200.0f;//exp constant
    float c = 0.75f;//angular v term constant
    float d = -1.0f;//term reward
                    
    //thresholds 
    double angularThX = 0.2;
    double angularThY = 0.2;



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
        

        //******RESET TRACKING VARIABLES*******
        t = 0.0; 
        timeSinceLastControl = controlDt; 
        timeSteps = 0;
        epTerminated = false;

        //******RESET VECTORS*******
        stateVector = {0.0, 0.0, 0.0, 0.0};
        stateQ = {1.0, 0.0, 0.0, 0.0};

        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0}; 
        angularAccleration = {0.0, 0.0, 0.0};
        angularVelocity = {0.0, 0.0, 0.0};
        angularPosition = {0.0, 0.0};
        
        //******RESET SERVOS******
        currentServoX = 0.0f;
        currentServoY = 0.0f;
    
        //******RESET STD VECTORS USED FOR STORAGE******
        y1Mat.clear();
        y2Mat.clear();
        y3Mat.clear();

        a1Mat.clear();
        a2Mat.clear();
        a3Mat.clear();

        stateVectorMat.clear();
        actionsMat.clear();
        rewardSeq.clear();


        //*****EPISODE GENERATION******
        for(int i = 0; i < (int)(simTime/dt); i++){
            //time increment
            t += dt;
            timeSinceLastControl += dt;

            //*****CONTROL*****
            if(timeSinceLastControl >= controlDt){
                timeSinceLastControl = 0.0;



                //*****REWARD*****
                //ensure only getting reward after initial action
                if(i > 0){
                    std::array<float, 2> rewardVec = reward(stateVector, a, b, c, d, angularThX, angularThY);//get reward and state of ep term

                    rewardSeq.push_back(rewardVec[0]);//log reward 

                    //check the ep state and update
                    if(rewardVec[1]){
                        epTerminated = true;
                        break;
                    }
                }



                //*****STATE*****
                stateVector = {angularPosition[0], angularPosition[1], angularVelocity[0], angularVelocity[1]};//create state 
                stateVectorMat.push_back(stateVector);//store state



                //*****MLP PROPAGATION*****
                mlpOutputTraining mlpOut = mlpOutTrain(stateVector, w1, w2, w3, b1, b2, b3);//get output

                //store outputs as is for hidden layers
                y1Mat.push_back(mlpOut.y_1);
                a1Mat.push_back(mlpOut.a_1);

                y2Mat.push_back(mlpOut.y_2);
                a2Mat.push_back(mlpOut.a_2);

                //store clamped outputs for output layer, for direct use in update loop
                //x outputs
                float meanX = mlpOut.y_out[0];
                float logSigmaXRaw = mlpOut.y_out[1];
                float logSigmaX = std::clamp(logSigmaXRaw, lowerBoundLogSigma, upperBoundLogSigma);
                
                //y outputs
                float meanY = mlpOut.y_out[2];
                float logSigmaYRaw = mlpOut.y_out[3];
                float logSigmaY = std::clamp(logSigmaYRaw, lowerBoundLogSigma, upperBoundLogSigma);

                y3Mat.push_back({meanX, logSigmaX, meanY, logSigmaY});



                //*******ACTIONS******
                //standard deviations
                float standardDevX = std::exp(logSigmaX);//e^log(sigma_x) = sigma_x                 
                float standardDevY = std::exp(logSigmaY);//e^log(sigma_y) = sigma_y               
                                                         
                std::normal_distribution<float> gaussianX(meanX, standardDevX);
                std::normal_distribution<float> gaussianY(meanY, standardDevY);

                float rawActionX = gaussianX(gen);
                float rawActionY = gaussianY(gen);

                //tanh squashing
                actionX = 0.087*std::tanh(rawActionX);//multiply by 0.087 rads(5 degs) such that scale up to be within action space
                actionY = 0.087*std::tanh(rawActionY);


                actionsMat.push_back({rawActionX, rawActionY});//log the actions(raw ones as denoted in notes from change of vars for pi_act)


                //******INCREMENT*******
                timeSteps++;
            }
           

            //*****UPDATE IN ENV*****
            //slew the servos to ensure only moving within physical limits
            slewServo(currentServoX, actionX, dt);//x slew
            slewServo(currentServoY, actionY, dt);//y slew
           

            //physics update
            stateUpdate(dt, t, U, sigmaU, currentServoX, currentServoY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularAccleration, angularVelocity, angularPosition);
        }

        //******ADD FLIGHT TIME TO SUM******
        timeSum += t;


        //******LOG FINAL REWARD IF EP. DID NOT END IN TERMINATION
        if(!epTerminated){
            //reward for otherwise 
            std::array<float, 2> rewardVec = reward(stateVector, a, b, c, d, angularThX, angularThY);
            rewardSeq.push_back(rewardVec[0]);//log it
        }


        //******VARS FOR UPDATE RULE LOOP******
        //vector that accumalted grad_t for each term in it, for whole loop in update rule
        std::array<float, 11004> accumlatedGrad = {0};

        //total discounted reward from t_0
        float totalEpisodeReturn = 0.0f;
    
        //clamp checks
        bool logSigmaXClamped = false;
        bool logSigmaYClamped = false;

        //nan check for a term in grad_t
        bool isNan = false;


        //******REINFORCE UPDATE LOOP FOR EPISODE*******
        for(int t = 0; t < timeSteps; t++){

            //*****RETURN_t*****
            returnT = 0.0f;

            //compute return at step t 
            for(int k = t; k < timeSteps; k++){
                returnT += std::pow(gamma, k-t)*rewardSeq.at(k);

            }
            if(t == 0) totalEpisodeReturn = returnT;//log the total discounted return from t_0

            //****GRADIENT TERM*****
            //gradLP_x
            float meanX = y3Mat[t][0];
            float logSigmaX = y3Mat[t][1];
            float standardDevX = std::exp(logSigmaX);
            float actionX = actionsMat[t][0];

            //clamping check for log sigma
            if(logSigmaX == lowerBoundLogSigma || logSigmaX == upperBoundLogSigma) logSigmaXClamped = true;
            else logSigmaXClamped = false;

            std::array<float, 11004> gradLogPolicy_x = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionX, meanX, standardDevX, 0, logSigmaXClamped);

            //gradLP_y
            float meanY = y3Mat[t][2];
            float logSigmaY = y3Mat[t][3];
            float standardDevY = std::exp(logSigmaY);
            float actionY = actionsMat[t][1];

            //clamping check for log sigma
            if(logSigmaY == lowerBoundLogSigma || logSigmaY == upperBoundLogSigma) logSigmaYClamped = true;
            else logSigmaYClamped = false;
            
            std::array<float, 11004> gradLogPolicy_y = gradLogPolicy_i(w2, w3, y1Mat.at(t), y2Mat.at(t), a1Mat.at(t), a2Mat.at(t), stateVectorMat.at(t), actionY, meanY, standardDevY, 2, logSigmaYClamped);
   

            //*****ASSEMBLE GRADIENT TERM AT t*****
            std::array<float, 11004> gradTerm_t = gradientTerm_t(gradLogPolicy_x, gradLogPolicy_y, returnT, t);


            //******ACCUMLATE GRAD TERM AT t******
            for(int i = 0; i < 11004; i++){
                //******NAN CHECK******
                isNan = std::isnan(gradTerm_t[i]);
                if(isNan){
                    std::cout << "gradTerm_t[" << i << "] = " << gradTerm_t[i] << " at time step " << t << std::endl;
                    break;
                }

                accumlatedGrad[i] += gradTerm_t[i];
            }

            if(isNan) break;//nan check
        }
        if(isNan) break;//nan check


        //******APPLY UPDATE TO PARAMETERS******
        std::array<float, 11004> gradTerm = gradientTerm(accumlatedGrad, alpha, gradientClipValue);
        updateParameters(gradTerm, w1, w2, w3, b1, b2, b3);

        
        //*****END OF EPISODE BATCH LOG******
        if((e % 1000 == 0 && e > 0) || e == numEpisodes-1){
            std::cout << "************************************************************" << std::endl;
            std::cout << "Total discounted return for episode " << e << " was G = " << totalEpisodeReturn << std::endl;
            std::cout << "Average flight time was: " << timeSum /(e*1.0) << "s" << std::endl;
        }
    }
    

    //*****WRITE PARAMETER BYTES TO .bin******
    writeParameters(w1, w2, w3, b1, b2, b3);


    return 0;
}
