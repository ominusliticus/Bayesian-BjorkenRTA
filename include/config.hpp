//
// Author: Kevin Ingles

// File for defining all the preprocessing commands for all other files


// Uncomment to enable parallel evaluation
// Parallelizing used for ExactSolution.cpp and BayesianParameterEstimation.hpp
#define USE_PARALLEL 0

// Need for dertain inversions of determinatn calculations
#define ARMA_DONT_USE_WRAPPER
#ifndef ARMA_USE_LAPACK
    #define ARMA_USE_LAPACK
#endif
