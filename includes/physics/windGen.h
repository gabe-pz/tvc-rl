#pragma once 

#include <cstdio>
#include <vector>
#include <random>  

//wind modeling global constants 
inline constexpr double ALPHA = 5.0/3.0;//spectral exponent where, S(f) ~ 1/f^alpha
inline constexpr int POLES = 2;//2 poles thus limiting freq to ~0.3 Hz, such that gusts shorter than ~3-5 s
inline constexpr double GEN_FREQ = 20.0;//fixed turbulence generation frequency 
inline constexpr double GEN_DT = 1.0/GEN_FREQ;//about 0.05 s between generated samples
inline constexpr double PINK_STD = 2.252;//standard deviation of a long 2 pole pink sequence

void computeCoefficients(double a[POLES + 1]);

double gaussianWhite(std::mt19937 &rng);
double windVelocity(double t, double U, double sigmaU, const std::vector<double> &pink);

std::vector<double> generatePinkNoise(int n, unsigned seed);
