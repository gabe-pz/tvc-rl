#include "windGen.h"

//******wind generation process used from the OpenRocket Techincal Documentation section 4.1.2*****


//filter coefficients via recursion of, a_k = (k - 1 - alpha/2) * a_{k-1} / k, a_0 = 1 
void computeCoefficients(double a[POLES + 1]){
    a[0] = 1.0;

    for (int k = 1; k <= POLES; k++){
        a[k] = (k-1.0-ALPHA/2.0)*a[k - 1] / k;//gives a1 = -5/6, a2 = -5/72
    }
                                                     
}


//one sample of the zero mean, unit variance Gaussian white noise 
double gaussianWhite(std::mt19937 &rng){

    std::normal_distribution<double> dist(0.0, 1.0);

    return dist(rng);
}

//generating n samples of unit std pink noise with the Kasdin IIR filter of, x_n = w_n - a1*x_{n-1} - a2*x_{n-2} (eq 4.5 truncated to 2 poles)  
std::vector<double> generatePinkNoise(int n, unsigned seed){

    double a[POLES + 1];
    computeCoefficients(a);

    std::mt19937 rng(seed);          
    std::vector<double> x(n);

    double x1 = 0.0;
    double x2 = 0.0;     

    for (int i = 0; i < n; i++){

        double w = gaussianWhite(rng);

        //the filter will drag past values along, where correlating samples causes boosts in low frequencies
        double xi = w-a[1]*x1-a[2]*x2;
        x2 = x1;
        x1 = xi;

        //rescale such that sequence has standard deviation of pretty much 1
        x[i] = xi/PINK_STD;       
    }
    return x;
}

//wind speed at arbitrary time t, is, w_speed = avgWindV+sigma_u*(interpolated pink sample)
double windVelocity(double t, double U, double sigmaU, const std::vector<double> &pink){

    int i = (int)(t/GEN_DT);//index of the 20 Hz sample just before 
                              
    if(i < 0){
        i = 0;
    }

    if(i > (int)pink.size()-2){
        i = pink.size()-2;
    }


    double frac = (t-i*GEN_DT)/GEN_DT;//fractional position between the samples, i and i+1 
    double x = (1.0-frac)*pink[i]+frac*pink[i + 1];//some inear interpolation action 
                                                   
    return(U+sigmaU*x);
}
