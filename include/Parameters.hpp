//
// Author: Kevin Ingles

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "Errors.hpp"

#include <vector>
#include <armadillo>

using vec = arma::vec;

struct SimulationParameters
{
    SimulationParameters() = default;
    SimulationParameters(const char* filename);
    ~SimulationParameters();

    void SetParameter(const char* name, double value);
    void SetParameters(double _tau_0, double _Lambda_0, double _xi_0, double _alpah_0, double _ul, double _mass, double _eta_s);
    void SetInitialTemperature(void);
    double IntegralJ(int n, int r, int q, int s, double mass, vec& X);

    friend std::ostream& operator<<(std::ostream& out, SimulationParameters& params);

    unsigned int type; // Which hydro simulation to compute

    double tau_0;
    double Lambda_0;
    double xi_0;
    double alpha_0;

    // For exact solution code - specific
    double ul; // Stopping time for simulation
    double ll; // Starting time for simulation 
    double mass;
    double eta_s;

    double steps;
    double step_size;

    // parameters for hydrodynamic evolution equations
    double pl0;
    double pt0;

    // TO DO: convert exact evolution into sturct an remove this vecotr, but keep initial value parameter
    double T0;
}; // end struct SimulationParameters

#endif