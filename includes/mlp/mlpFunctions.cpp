#include "mlpFunctions.h"


//*****PARAMETERS INIT*****
void initParameters(std::array<std::array<float, 6>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3){
   
    
    //random num
    std::random_device rd; 
    std::mt19937 gen(rd());

    //distribution populate weights from
    std::uniform_real_distribution<float> dist(-0.77, 0.77);//number from formula of x = sqrt(6/(n_in+n_out)) 
                                                               
    //init w1 
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 6; j++){
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
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            w1[i][j] = dist(gen);
        }
    }

    //init biases to zero 
    b1.fill(0.0f);
    b2.fill(0.0f);
    b1.fill(0.0f);
}



//*****ACTIVATION FUNCTION*****
float ReLU(float y){
    if(y > 0){
        return y;
    }
    else{
        return 0;
    }
}



//*****Pre-activations and activations*****
template<typename T, std::size_t rows, std::size_t cols>//rows and cols of the weight matrix for current layer
std::array<float, rows> preActivationVec(const std::array<std::array<float, cols>, rows>& weightMatrix, const std::array<T, cols>& vecIn, const std::array<float, rows>& biasVec){

    //init preActivation vector 
    std::array<float, rows> y_N = {0}; 

    //weight matrix multipled by input vector
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            y_N[i] = y_N[i] + weightMatrix[i][j]*vecIn[j];
        }
    }

    //add bias to each term
    for(int i = 0; i < rows; i++){
        y_N[i] = y_N[i] + biasVec[i];
    }

    return y_N;
}


template<std::size_t N>//size of the pre-activation vector 
std::array<float, N> activationsVec(const std::array<float, N>& y_N){

    std::array<float, N> act_N;

    for(int i = 0; i < N; i++){
        act_N[i] = ReLU(y_N[i]);
    }

    return act_N;
}



