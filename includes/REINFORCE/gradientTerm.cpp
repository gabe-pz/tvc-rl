#include "gradientTerm.h"
#include "gradientParts.h"


std::array<float, 12204> gradLogPolicy_i(const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& y1, 
        const std::array<float, 100>& y2, const std::array<double, 6>& stateVector, float action_i, float mean_i, float standardDev_i, int meanIndex){
    std::array<float, 12204> gradLP;

    float term_i_1 = (action_i - mean_i) / (standardDev_i*standardDev_i);
    float term_i_2 = ((action_i-mean_i)*(action_i-mean_i) - standardDev_i*standardDev_i) / (standardDev_i*standardDev_i);
    std::array<float, 2> terms_i = {term_i_2, term_i_2};

    std::array<float, 600> gLP1 = gradientLogPolicy_1(w2, w3, y2, stateVector, terms_i, meanIndex);
    std::array<float, 100> gLP2 = gradientLogPolicy_2(w2, w3, y2, terms_i, meanIndex);

    std::array<float, 10000> gLP3 = gradientLogPolicy_3(w3, y2, y1, terms_i, meanIndex);
    std::array<float, 100> gLP4 = gradientLogPolicy_4(w3, y2, terms_i, meanIndex);

    std::array<float, 400> gLP5 = gradientLogPolicy_5(y2, terms_i); 
    std::array<float, 4> gLP6 = gradientLogPolicy_6(terms_i);

    return gradLP; 
}
