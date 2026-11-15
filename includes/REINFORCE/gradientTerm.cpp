#include "gradientTerm.h"
    
std::array<float, 12204> assembleGragLogPolicy(const std::array<float, 600>& gLP1, const std::array<float, 100>& gLP2, const std::array<float, 10000>& gLP3, 
        const std::array<float, 100>& gLP4, const std::array<float, 400>& gLP5, const std::array<float, 4>& gLP6){
    std::array<float, 12204> gLP; 
    int count = 0;

    for(int i = 0; i < 600; i++){
        gLP[count] = gLP1[i];
        count++;
    }
    for(int i = 0; i < 100; i ++){
        gLP[count] = gLP2[i];
        count ++;
    }

    for(int i = 0; i < 10000; i++){
        gLP[count] = gLP3[i];
        count++;
    }
    for(int i = 0; i < 100; i ++){
        gLP[count] = gLP4[i];
        count ++;
    }
    for(int i = 0; i < 400; i++){
        gLP[count] = gLP5[i];
        count++;
    }
    for(int i = 0; i < 4; i ++){
        gLP[count] = gLP6[i];
        count ++;
    }

    return gLP;
}

std::array<float, 12204> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVector, float action_i, float mean_i, float standardDev_i, int meanIndex){

    float term_i_1 = (action_i - mean_i) / (standardDev_i*standardDev_i);
    float term_i_2 = ((action_i-mean_i)*(action_i-mean_i) - standardDev_i*standardDev_i) / (standardDev_i*standardDev_i);
    std::array<float, 2> terms_i = {term_i_2, term_i_2};

    std::array<float, 600> gLP1 = gradientLogPolicy_1(w2, w3, y2, stateVector, terms_i, meanIndex);
    std::array<float, 100> gLP2 = gradientLogPolicy_2(w2, w3, y2, terms_i, meanIndex);

    std::array<float, 10000> gLP3 = gradientLogPolicy_3(w3, y2, y1, terms_i, meanIndex);
    std::array<float, 100> gLP4 = gradientLogPolicy_4(w3, y2, terms_i, meanIndex);

    std::array<float, 400> gLP5 = gradientLogPolicy_5(y2, terms_i); 
    std::array<float, 4> gLP6 = gradientLogPolicy_6(terms_i);

    std::array<float, 12204> gradLP = assembleGragLogPolicy(gLP1, gLP2, gLP3, gLP4, gLP5, gLP6);
    return gradLP; 
}


