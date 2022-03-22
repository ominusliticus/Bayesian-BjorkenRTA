//
// Author: Kevin Ingles

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "Errors.hpp"

#include <vector>
#include <armadillo>
#include <string>

using vec = arma::vec;

struct SimulationParameters
{
    SimulationParameters() = default;
    SimulationParameters(const char* filename);
    ~SimulationParameters();

    static SimulationParameters ParseCmdLine(int cmdln_count, char** cmdln_args);
    void SetParameter(const char* name, double value);
    void SetParameters(double _tau_0, double _Lambda_0, double _xi_0, double _alpah_0, double _tau_f, double _mass, double _eta_s);
    void SetInitialTemperature(void);
    double IntegralJ(int n, int r, int q, int s, double mass, vec& X);

    friend std::ostream& operator<<(std::ostream& out, SimulationParameters& params);
    bool operator==(const SimulationParameters& other);
    bool operator!=(const SimulationParameters& other);

    unsigned int type; // Which hydro simulation to compute

    double tau_0;
    double Lambda_0;
    double xi_0;
    double alpha_0;

    double tau_f; 
    double mass;
    double C;

    double steps;
    double step_size;

    // parameters for hydrodynamic evolution equations
    double pl0;
    double pt0;

    // TO DO: convert exact evolution into sturct an remove this vecotr, but keep initial value parameter
    double T0;

    std::string file_identifier;
}; // end struct SimulationParameters

#endif
