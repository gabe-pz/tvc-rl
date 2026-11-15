#include "gradientTerm.h"
#include "derivatives.h"



std::array<float, 600> gradientLogPolicy_1(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3,
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

        for(int j = 0; j < 6; j++){
            float derivativeMu = derivativeActWrtPreAct(y2, i)*derivativeActWrtWeight(stateVec, j)*sumTermMu;
            float derivativeLogSigma = derivativeActWrtPreAct(y2, i)*derivativeActWrtWeight(stateVec, j)*sumTermLogSigma;

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

std::array<float, 100> gradientLogPolicy_2(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3,
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


        float derivativeMu = derivativeActWrtPreAct(y2, i)*sumTermMu;
        float derivativeLogSigma = derivativeActWrtPreAct(y2, i)*sumTermLogSigma;

        derivativeTermsMu[i] = derivativeMu;
        derivativeTermsLogSigma[i] = derivativeLogSigma;
    }


    //*****ASSEMBLE PART 1 OF GRADIENT*****
    for(int i = 0; i < 100; i++){
        gLP2[i] = terms_i[0]*derivativeTermsMu[i] + terms_i[1]*derivativeTermsLogSigma[i];
    }

    return gLP2;
}



