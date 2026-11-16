#include "gradientParts.h"


//*****DERIVATIVE FUNCTIONS*******
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
float derivativePreActWrtWeight(const std::array<T, N>& inputVector, int j){
    return inputVector.at(j);//j'ith element in inputVector, where j is the col of weight matrix corresponding to particular weight taking derivative wrt
}


//******GRADIENT PARTS******
std::array<float, 600> gradientLogPolicy_1(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1,
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVec, const std::array<float, 2>& terms_i, int meanIndex){

    std::array<float, 600> gLP1;

    std::array<float, 600> derivativeTermsMu;
    std::array<float, 600> derivativeTermsLogSigma;

    //*****TERMS FOR DERIVATIVE*****
    for(int i = 0; i < 100; i++){

        //sum term
        float sumTermMu = 0.0f;
        float sumTermLogSigma = 0.0f;

        for(int k = 0; k < 100; k++){
            float sigmaTermMu_k = derivativePreActWrtAct(w3, meanIndex, k)*derivativeActWrtPreAct(y2, k)*derivativePreActWrtAct(w2, k, i);
            float sigmaTermLogSigma_k = derivativePreActWrtAct(w3, meanIndex+1, k)*derivativeActWrtPreAct(y2, k)*derivativePreActWrtAct(w2, k, i);

            sumTermMu += sigmaTermMu_k;
            sumTermLogSigma += sigmaTermLogSigma_k;
        }

        //vectors for terms for the derivatives of outputs
        for(int j = 0; j < 6; j++){
            float derivativeMu = derivativeActWrtPreAct(y1, i)*derivativePreActWrtWeight(stateVec, j)*sumTermMu;
            float derivativeLogSigma = derivativeActWrtPreAct(y1, i)*derivativePreActWrtWeight(stateVec, j)*sumTermLogSigma;

            derivativeTermsMu[6*i+j] = derivativeMu;
            derivativeTermsLogSigma[6*i+j] = derivativeLogSigma;
        }
    }

    //*****ASSEMBLE PART 1 OF GRADIENT*****
    for(int i = 0; i < 600; i++){
        gLP1[i] = terms_i[0]*derivativeTermsMu[i] + terms_i[1]*derivativeTermsLogSigma[i];
    }

    return gLP1;
}

std::array<float, 100> gradientLogPolicy_2(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1,
        const std::array<float, 100>& y2, const std::array<float, 2>& terms_i, int meanIndex){

    std::array<float, 100> gLP2; 

    std::array<float, 100> derivativeTermsMu;
    std::array<float, 100> derivativeTermsLogSigma;

    for(int i = 0; i < 100; i++){
        
        //sum terms 
        float sumTermMu = 0.0f;
        float sumTermLogSigma = 0.0f;

        for(int k = 0; k < 100; k++){

            float sigmaTermMu_k = derivativePreActWrtAct(w3, meanIndex, k)*derivativeActWrtPreAct(y2, k)*derivativePreActWrtAct(w2, k, i);
            float sigmaTermLogSigma_k = derivativePreActWrtAct(w3, meanIndex+1, k)*derivativeActWrtPreAct(y2, k)*derivativePreActWrtAct(w2, k, i);

            sumTermMu += sigmaTermMu_k;
            sumTermLogSigma += sigmaTermLogSigma_k;
        }

        //vectors for terms for the derivatives of outputs
        float derivativeMu = derivativeActWrtPreAct(y1, i)*sumTermMu;
        float derivativeLogSigma = derivativeActWrtPreAct(y1, i)*sumTermLogSigma;

        derivativeTermsMu[i] = derivativeMu;
        derivativeTermsLogSigma[i] = derivativeLogSigma;
    }


    //*****ASSEMBLE PART 2 OF GRADIENT*****
    for(int i = 0; i < 100; i++){
        gLP2[i] = terms_i[0]*derivativeTermsMu[i] + terms_i[1]*derivativeTermsLogSigma[i];
    }

    return gLP2;
}


std::array<float, 10000> gradientLogPolicy_3(const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y2, const std::array<float, 100>& a1, 
        const std::array<float, 2>& terms_i, int meanIndex){

    std::array<float, 10000> gLP3;

    std::array<float, 10000> derivativeTermsMu;
    std::array<float, 10000> derivativeTermsLogSigma;

    for(int i = 0; i < 100; i++){
        float derivativeTerm1Mu = derivativePreActWrtAct(w3, meanIndex, i);
        float derivativeTerm1LogSigma = derivativePreActWrtAct(w3, meanIndex+1, i);

        float derivativeTerm2 = derivativeActWrtPreAct(y2, i);

        for(int j = 0; j < 100; j++){
            float derivativeTerm3 = derivativePreActWrtWeight(a1, j);

            derivativeTermsMu[100*i+j] = derivativeTerm1Mu*derivativeTerm2*derivativeTerm3;
            derivativeTermsLogSigma[100*i+j] = derivativeTerm1LogSigma*derivativeTerm2*derivativeTerm3;
        }
    }

    for(int i = 0; i < 10000; i++){
        gLP3[i] = terms_i[0]*derivativeTermsMu[i] + terms_i[1]*derivativeTermsLogSigma[i];
    }

    return gLP3;
}

std::array<float, 100> gradientLogPolicy_4(const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y2, const std::array<float, 2>& terms_i, int meanIndex){

    std::array<float, 100> gLP4;

    std::array<float, 100> derivativeTermsMu;
    std::array<float, 100> derivativeTermsLogSigma;

    for(int i = 0; i < 100; i++){
        float derivativeTerm1Mu = derivativePreActWrtAct(w3, meanIndex, i);
        float derivativeTerm1LogSigma = derivativePreActWrtAct(w3, meanIndex+1, i);

        float derivativeTerm2 = derivativeActWrtPreAct(y2, i);


        derivativeTermsMu[i] = derivativeTerm1Mu*derivativeTerm2;
        derivativeTermsLogSigma[i] = derivativeTerm1LogSigma*derivativeTerm2;
    }

    for(int i = 0; i < 100; i++){
        gLP4[i] = terms_i[0]*derivativeTermsMu[i] + terms_i[1]*derivativeTermsLogSigma[i];
    }

    return gLP4;
}

std::array<float, 400> gradientLogPolicy_5(const std::array<float, 100>& a2, const std::array<float, 2>& terms_i, int meanIndex){
    std::array<float, 400> gLP5 = {0};

    //determine if outputs corresponding to x or y
    if(meanIndex == 0){

        for(int i = meanIndex; i < meanIndex+2; i++){

            for(int j = 0; j < 100; j++){
                if(i == meanIndex) gLP5[100*i+j] = terms_i[0]*derivativePreActWrtWeight(a2, j);
                else gLP5[100*i+j] = terms_i[1]*derivativePreActWrtWeight(a2, j);
            }

        }
    }
    else{

        for(int i = meanIndex; i < meanIndex+2; i++){

            for(int j = 0; j < 100; j++){
                if(i == meanIndex) gLP5[100*i+j] = terms_i[0]*derivativePreActWrtWeight(a2, j);
                else gLP5[100*i+j] = terms_i[1]*derivativePreActWrtWeight(a2, j);
            }

        }
    }


    return gLP5;
}
std::array<float, 4> gradientLogPolicy_6(const std::array<float, 2>& terms_i, int meanIndex){

    //determine if outputs corresponding to x or y
    //x
    if(meanIndex == 0){
        std::array<float, 4> gLP6 = {terms_i[0], terms_i[1], 0.0f, 0.0f};
        return gLP6;
    }

    //y
    else{
        std::array<float, 4> gLP6 = {0.0f, 0.0f, terms_i[0], terms_i[1]};
        return gLP6;
    }
}
