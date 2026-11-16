#include "gradientTerm.h"
    
std::array<float, 11204> assembleGragLogPolicy(const std::array<float, 600>& gLP1, const std::array<float, 100>& gLP2, const std::array<float, 10000>& gLP3, 
        const std::array<float, 100>& gLP4, const std::array<float, 400>& gLP5, const std::array<float, 4>& gLP6){
    std::array<float, 11204> gLP; 
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

std::array<float, 11204> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<float, 100>& a1, const std::array<float, 100>& a2, const std::array<double, 6>& stateVector,
        float action_i, float mean_i, float standardDev_i, int meanIndex){

    float term_i_1 = (action_i - mean_i) / (standardDev_i*standardDev_i);
    float term_i_2 = ((action_i-mean_i)*(action_i-mean_i) - standardDev_i*standardDev_i) / (standardDev_i*standardDev_i);

    std::array<float, 2> terms_i = {term_i_1, term_i_2};

    std::array<float, 600> gLP1 = gradientLogPolicy_1(w2, w3, y1, y2, stateVector, terms_i, meanIndex);
    std::array<float, 100> gLP2 = gradientLogPolicy_2(w2, w3, y1, y2, terms_i, meanIndex);
    
    std::array<float, 10000> gLP3 = gradientLogPolicy_3(w3, y2, a1, terms_i, meanIndex);
    std::array<float, 100> gLP4 = gradientLogPolicy_4(w3, y2, terms_i, meanIndex);

    std::array<float, 400> gLP5 = gradientLogPolicy_5(a2, terms_i, meanIndex);
    std::array<float, 4> gLP6 = gradientLogPolicy_6(terms_i, meanIndex);

    std::array<float, 11204> gradLP = assembleGragLogPolicy(gLP1, gLP2, gLP3, gLP4, gLP5, gLP6);
    return gradLP; 
}


std::array<float, 11204> gradientTerm(const std::array<float, 11204>& gradLPx, const std::array<float, 11204>& gradLPy, float returnT, float alpha, float gamma, int t){
    std::array<float, 11204> gradTerm;
    
    //add grad log pi x and grad log pi y together
    for(int i = 0; i < 11204; i++){
        float delta_t = alpha*returnT*std::pow(gamma, t);
        gradTerm[i] = delta_t*(gradLPx[i]+gradLPy[i]);
    }
    
    return gradTerm;
}


void updateParameters(const std::array<float, 11204>& accumaltedGradTerm, std::array<std::array<float, 6>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, 
        std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, std::array<float, 100>& b2, std::array<float, 4>& b3){
    
    int index = 0;

    //update w1 
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 6; j++){
            w1[i][j] += accumaltedGradTerm[index];
            index++;
        }
    }
    //update b1
    for(int i = 0; i < 100; i++){
        b1[i] += accumaltedGradTerm[index];
        index++;
    }

    //update w2
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            w2[i][j] += accumaltedGradTerm[index];
            index++;
        }
    }
    //update b2
    for(int i = 0; i < 100; i++){
        b2[i] += accumaltedGradTerm[index];
        index++;
    }

    //update w3
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            w3[i][j] += accumaltedGradTerm[index];
            index++;
        }
    }

    //update b3
    for(int i = 0; i < 4; i++){
        b3[i] += accumaltedGradTerm[index];
        index++;
    }
}
