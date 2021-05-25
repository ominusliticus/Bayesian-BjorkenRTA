#pragma once
#include "Errors.hpp"

#include <vector>

struct SimulationParameters
{
    SimulationParameters(const char* filename);
    ~SimulationParameters();

    friend std::ostream& operator<<(std::ostream& out, SimulationParameters& params);

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

    std::vector<double> D;
}; // end struct SimulationParameters