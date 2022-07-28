//  Copyright 2021-2022 Kevin Ingles
//  
//  Permission is hereby granted, free of charge, to any person obtaining
//  a copy of this software and associated documentation files (the
//  "Software"), to deal in the Software without restriction, including
//  without limitation the right to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of the Software, and to
//  permit persons to whom the Sofware is furnished to do so, subject to
//  the following conditions:
//  
//  The above copyright notice and this permission notice shall be
//  included in all copies or substantial poritions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//  SOFTWARE OR THE USE OF OTHER DEALINGS IN THE SOFTWARE
// 
// Author: Kevin Ingles
// Credit: Chandrodoy Chattopadhyay for original code
// File: ExactSolution.hpp
// Description: Header file which defines the routines and data structures to 
//              facilitate the calculation of the Boltzmann RTA exact solution 
//              for Bjorken expansion

#ifndef EXACT_SOLUTION_HPP
#define EXACT_SOLUTION_HPP

#include <vector>
#include <tuple>

#include "config.hpp"
#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"

// For parallelizing the temperature evolution loops in eaxt::ExactSolution::Run()
#if USE_PARALLEL
    #include <omp.h>
#endif

using SP = SimulationParameters;

// TODO: Add documentation to comments that explain what all the variables used mean (can be done in implementation file)

namespace exact{
    // Enum to indicate which moment to calculate
    // ED - Energy density
    // PL - Longitudinal pressure
    // PT - Transvers pressure
    // PEQ - Equilibrium pressure
    enum class Moment { ED = 0, PL, PT, PEQ };

    ////////////////////////////////////////////////////////////
    ///        Code need by all subsequent functions         ///
    ////////////////////////////////////////////////////////////

    struct ExactSolution
    {
        ExactSolution() = default;

        
        // Get temperature based current time step
        double GetTemperature(double z, SP& params);
        // Calculate relaxation time given proper time tau
        double TauRelaxation(double tau, SP& params);
        // Exponential of integral of inverse relaxation time
        double DecayFactor(double tau1, double tau2, SP& params);


        ////////////////////////////////////////////////////////////
        ///        Code for evaluating moments analytically      ///
        ////////////////////////////////////////////////////////////

        // Special functions: for calculating analytic result of exact distribution function
        double H(double alpha, double zeta, Moment flag);
        double HTilde(double y, double z, Moment flag);
        double HTildeAux(double u, double y, double z, Moment flag);

        // Analytic results for moments of exact solution
        // Equilibrium energy density (momentum integration already)
        double EquilibtirumDistributionMoment(double tau, double z, SP& params, Moment flag);
        // Energy density from initial distribution function
        double InitialDistributionMoment(double tau, SP& params, Moment flag);

        // The integral over the initial distribution founction for exact solution
        double EquilibriumContribution(double tau, SP& params, Moment flag);
        double EquilibriumContributionAux(double x, double tau, SP& params, Moment flag);

        // Returns nth moment for particle distribution function
        double GetMoments(double tau, SP& params, Moment flag);


        ////////////////////////////////////////////////////////////
        ///        Code for evaluating moments numerically       ///
        ////////////////////////////////////////////////////////////

        // TO DO: Introduce different parameteizations
        double InitialDistribution(double w, double pT, SP& params);
        double EquilibriumDistribution(double w, double pT, double tau, SP& params);

        // exact solution to Boltzmann equation in RTA
        double EaxctDistribution(double w, double pT, double tau, SP& params);
        double ThetaIntegratedExactDistribution(double p, double tau, SP& params);

        // Evaluating moments of distribution function numerically
        double GetMoments2(double tau, SP& params, Moment flag);
        double GetMoments2Aux(double pT, double tau, SP& params, Moment flag);

        // Only necessary for debuggin to be deleted
        std::tuple<double, double> EaxctDistributionTuple(double w, double pT, double tau, SP& params);
        std::tuple<double, double> ThetaIntegratedExactDistributionTuple(double p, double tau, SP& params);


        ////////////////////////////////////////////////////////////
        ///        Code to solve temperature evolution           ///
        ////////////////////////////////////////////////////////////

        // Calculate the equilibrium energy density at temperature T_eq 
        // with momemtum integration
        double EquilibriumEnergyDensity(double temp, SP& params);

        // Return temperature corresponding to a given energy density e
        double InvertEnergyDensity(double e, SP& params);

        // Evolve simulation and output results;
        void Run(SP& params);

        // output moments of solution
        void OutputMoments(const char* file_name, SP& params);

    };
}

#endif
