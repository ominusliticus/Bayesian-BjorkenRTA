// 
// Author: Kevin Ingles
// Credit: Chandrodoy Chattopadhyay for original code

#ifndef EXACT_SOLUTION_HPP
#define EXACT_SOLUTION_HPP

#include <vector>
#include <tuple>

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"

using SP = SimulationParameters;

// TODO: Add documentation to comments that explain what all the variables used mean (can be done in implementation file)

namespace exact{
    enum class Moment { ED = 0, PL, PT };

    // Calculate the equilibrium energy density at temperature T_eq 
    // with momemtum integration
    double EquilibriumEnergyDensity(double temp, SP& params);
    double EquilibriumEnergyDensityAux(double p, double temp, SP& params);

    // Get temperature based current time step
    double GetTemperature(double z, SP& params);
    double InvertEnergyDensity(double e, SP& params);

    // Calculate relaxation time given proper time tau
    double TauRelaxation(double tau, SP& params);

    // Special function 
    double H(double alpha);
    double H2(double alpha, double zeta);
    double H2Tilde(double y, double z);
    double H2TildeAux(double u, double y, double z);

    // Equilibrium energy density (momentum integration already)
    double EquilibtirumEnergyDensity(double tau, double z, SP& params);

    // Initial distribution: will be able to introduce different parameteizations
    double InitialDistribution(double w, double pT, SP& params);
    // Equilibrium distribution
    double EquilibriumDistribution(double w, double pT, double tau, SP& params);

    // Exponential of integral of inverse relaxation time
    double DecayFactor(double tau1, double tau2, SP& params);

    // The integral over the initial distribution founction for exact solution
    double EquilibriumContribution(double tau, SP& params);
    double EquilibriumContributionAux(double x, double tau, SP& params);

    // Energy density from initial distribution function
    double InitialEnergyDensity(double tau, SP& params);

    // exact solution for distribution function
    double EaxctDistribution(double w, double pT, double tau, SP& params);
    double ThetaIntegratedExactDistribution(double p, double tau, SP& params);
    std::tuple<double, double> EaxctDistributionTuple(double w, double pT, double tau, SP& params);
    std::tuple<double, double> ThetaIntegratedExactDistributionTuple(double p, double tau, SP& params);

    // Returns nth moment for particle distribution function
    // Using analytic expressions
    double GetMoments(double tau, SP& params);
    // Doing momenta integrals
    double GetMoments2(double tau, SP& params, Moment flag);
    double GetMoments2Aux(double pT, double tau, SP& params, Moment flag);

    // Evolve simulation and output results;
    void Run(std::ostream& out, SP& params);
}

#endif