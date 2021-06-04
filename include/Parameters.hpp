//
// Author: Kevin Ingles

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "Errors.hpp"

#include <vector>

struct SimulationParameters
{
    SimulationParameters() = default;
    SimulationParameters(const char* filename);
    ~SimulationParameters();

    friend std::ostream& operator<<(std::ostream& out, SimulationParameters& params);

    double tau_0;     // Starting time
    double Lambda_0;  // Initial anisotropic temperature
    double xi_0;      // Initial anisotropy parameter
    double alpha_0;   // Initial bulk pressure parameter

    // For exact solution code - specific
    double ul; // Stopping time for simulation
    double ll; // Starting time for simulation 
    double mass;
    double eta_s;

    double steps;
    double step_size;

    std::vector<double> D;
}; // end struct SimulationParameters

#endif