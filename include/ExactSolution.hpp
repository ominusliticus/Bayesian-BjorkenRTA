#pragma once
#include <vector>

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"

using SP = SimulationParameters;

namespace exact{

    // Calculate the equilibrium energy density at temperature T_eq
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

    // Equilibrium distribution function
    double EquilibriumDistribution(double tau, double z, SP& params);

    // Initial distribution: will be able to introduce different parameteizations
    double InitialDistribution(double theta, double p, double tau, SP& params);

    // Exponential of integral of inverse relaxation time
    double DecayFactor(double tau1, double tau2, SP& params);

    // The integral over the initial distribution founction for exact solution
    double EquilibriumContribution(double tau, SP& params);
    double EquilibriumContributionAux(double x, double tau, SP& params);

    // Energy density from initial distribution function
    double InitialEnergyDensity(double tau, SP& params);

    // Returns nth moment for particle distribution function, needs modification...
    double GetMoments(double tau, SP& params);

    // Evolve simulation and output results;
    void Run(std::ostream& out, SP& params);
}