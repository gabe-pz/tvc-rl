#include "mlpFunctions.h"


void initParameters(std::array<std::array<float, 4>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3){
   
    
    //random num
    std::random_device rd; 
    std::mt19937 gen(rd());

    //distribution populate weights from
    std::uniform_real_distribution<float> dist(-0.77, 0.77);//number from formula of x = sqrt(6/(n_in+n_out)) 
                                                               
    //init w1 
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            w1[i][j] = dist(gen);
        }
    }

    //init w2
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            w1[i][j] = dist(gen);
        }
    }

    //init w3
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 4; j++){
            w1[i][j] = dist(gen);
        }
    }

    //init biases to zero 
    b1.fill(0.0f);
    b2.fill(0.0f);
    b1.fill(0.0f);
}


float ReLU(float y){
    if(y > 0){
        return y;
    }
    else{
        return 0;
    }
}


//*****Pre-activations and activations*****
std::array<float, 100> preActivation_1(const std::array<std::array<float, 4>, 100>& w1, const std::array<double, 6>& stateVector, const std::array<float, 100>& b1){

    //init preActivation vector 
    std::array<float, 100> y_1 = {0}; 

    //weight matrix multipled by state vector
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            y_1[i] = y_1[i] + w1[i][j]*stateVector[j];
        }
    }

    //add bias to each term
    for(int i = 0; i < 100; i++)
        y_1[i] = y_1[i] + b1[i];
    }

    return y_1;
}

