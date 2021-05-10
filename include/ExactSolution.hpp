#pragma once
#include <iostream>
#include <vector>

#include "Errors.hpp"
#include "Global_Constants.hpp"

namespace exact{
    class ExactSolution
    {
    public:
        ExactSolution() = default;
        ExactSolution(double tau_0, double Lambda_0, double xi_0, double ul, double ll, double mass, double eta_s, double steps, double step_size)
            : _tau_0(tau_0), _Lambda_0(Lambda_0), _xi_0(xi_0), _ul(ul), _ll(ll), _mass(mass), _eta_s(eta_s), _steps(steps), _step_size(step_size)
        {
            _D.resize(steps);
        }

        ~ExactSolution();

        // Calculate the equilibrium energy density at temperature T_eq
        double EquilibriumEnergyDensity(double temp);
        double EquilibriumEnergyDensityAux1(double p, double temp, double mass);
        double EquilibriumEnergyDensityAux2(double theta, double p, double temp, double mass);

        // Get temperature based current time step ??
        double GetTemperature(double z);
        double InvertEnergyDensity(double e);

        // Calculate relaxation time given proper time tau
        double TauRelaxation(double tau);

        // Special function 
        double H(double alpha);
        double H2(double alpha, double zeta);
        double H2Tilde(double y, double z);

        // Equilibrium distribution function: is this right???
        double EquilibriumDistribution(double tau, double z);

        // Initial distribution: will be able to introduce different parameteizations
        double InitialDistribution(double theta, double p, double tau, double mass);

        // Exponential of integral of inverse relaxation time
        double DecayFactor(double tau1, double tau2);

        // The integral over the initial distribution founction for exact solution
        double IntegralOfEquilibriumContribution(double tau);
        double IntegralOfEquilibriumContributionAux(double x, double tau, double mass);

        // Energy density from initial distribution function
        double InitialEnergyDensity(double tau);
        double InitialEnergyDensityAux(double p, double tau, double mass);

        // Returns nth moment for particle distribution function, needs modification...
        double GetMoments(double tau);

        // Evolve simulation and output results;
        void Run(int iters, std::ostream& out);


        // Initial conditions
        double _tau_0;
        double _Lambda_0;
        double _xi_0;

        
        double _ul; // what is this
        double _ll; // what is this
        double _mass;
        double _eta_s;

        // simulation specifics
        int _steps;
        double _step_size;
        std::vector<double> _D; // what is this

        // friend declarations to allow for
    };
}