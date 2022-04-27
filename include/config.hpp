//
// Author: Kevin Ingles
// File: config.hpp
// Description: For defining macros for different platforms and run configurations

#ifndef CONFIG_HPP
#define CONFIG_HPP

// Uncomment to enable parallel evaluation
// Parallelizing used for ExactSolution.cpp and BayesianParameterEstimation.hpp
#if __APPLE__
// My apple machine currently only supports single core processing
#define USE_PARALLEL 0
#else
#define USE_PARALLEL 1
#endif

// Need for dertain inversions of determinatn calculations
#define ARMA_DONT_USE_WRAPPER
#ifndef ARMA_USE_LAPACK
    #define ARMA_USE_LAPACK
#endif

#endif