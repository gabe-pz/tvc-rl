#include "derivatives.h"


template<std::size_t N>
float derivativeActWrtPreAct(const std::array<float, N>& preActVector, int n){
    if(preActVector.at(n) > 0){
        return 1.0f;
    }
    else return 0.0f;
}


template<std::size_t rows, std::size_t cols>//rows and cols of the weight matrix for current layer
float derivativePreActWrtAct(const std::array<std::array<float, cols>, rows>& wMat, int n, int m){
    return wMat.at(n).at(m);//nth row, mth col, where n is the element of preActivationVec taking derivative of, that is y_ln, and m is the element in activation vector taking derivative wrt, that is a_(l-1)m
}


template<typename T, std::size_t N>
float derivativeActWrtWeight(const std::array<T, N>& inputVector, int j){
    return inputVector.at(j);//j'ith element in inputVector, where j is the col of weight matrix corresponding to particular weight taking derivative wrt
}
