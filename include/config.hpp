//
// Author: Kevin Ingles

// File for defining all the preprocessing commands for all other files

#ifndef CONFIG_HPP
#define CONFIG_HPP

// Uncomment to enable parallel evaluation
// Parallelizing used for ExactSolution.cpp and BayesianParameterEstimation.hpp
#define USE_PARALLEL 1

// Flag that controls whether to use Armadillo for matrix operations or arrays
#define USE_ARMADILLO 1

// Need for dertain inversions of determinatn calculations
#if USE_ARMADILLO
    #define ARMA_DONT_USE_WRAPPER
    #ifndef ARMA_USE_LAPACK
        #define ARMA_USE_LAPACK
    #endif
#endif 

// Set for printing output in hydro
#define HYDRO_THEORIES_PRINT_OUTPUT 0


#endif
