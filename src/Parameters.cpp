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
// File: Parameters.cpp
// Description: This file implements the parameter parser, parsing either command line
//              arguments of input configuration files to run the hydrodynamic simulations

#include "Parameters.hpp"
#include "GlobalConstants.hpp"
#include "HydroTheories.hpp"
#include "Integration.hpp"
#include "InvertObservables.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

SimulationParameters::SimulationParameters(const char* filename)
{
    std::fstream fin(filename, std::fstream::in);
    if (!fin.is_open())
    {
        Print_Error(std::cerr, "Failed to open file: ", filename);
        exit(-2);
    }    // end if (!fin.is_open())

    const char        hash    = '#';
    const char        endline = '\0';
    std::string       line;
    std::stringstream buffer;
    std::string       var_name;
    while (!fin.eof())
    {
        std::getline(fin, line);
        if (line[0] == hash || line[0] == endline) continue;
        else
        {
            buffer = std::stringstream(line);
            // Note: assumes tab or space separation
            buffer >> var_name;
            if (var_name.compare("tau_0") == 0) buffer >> tau_0;
            else if (var_name.compare("tau_f") == 0) buffer >> tau_f;
            else if (var_name.compare("e0") == 0) buffer >> e0;
            else if (var_name.compare("pt0") == 0) buffer >> pt0;
            else if (var_name.compare("pl0") == 0) buffer >> pl0;
            else if (var_name.compare("mass") == 0) buffer >> mass;
            else if (var_name.compare("C") == 0) buffer >> C;
            else if (var_name.compare("steps") == 0) buffer >> steps;
            else if (var_name.compare("TYPE") == 0) buffer >> type;
            else if (var_name.compare("FILE") == 0) buffer >> file_identifier;
        }    // end else
    }        // end while(!fin.eof())
    step_size = tau_0 / 20;
    steps     = std::ceil((tau_f - tau_0) / step_size);

    if (this->type == 3 || this->type == 4) SetAnisotropicVariables();
    SetInitialTemperature();
    fin.close();
}    // end SimulationParameters::SimulationParameters(...)

// -----------------------------------------

SimulationParameters::~SimulationParameters()
{
}

SimulationParameters SimulationParameters::ParseCmdLine(int cmdln_count, char** cmdln_args)
{
    if (cmdln_count == 0) return SimulationParameters("utils/params.txt");
    SimulationParameters params{};
    for (int i = 1; i < cmdln_count - 1; i += 2)
        params.SetParameter(cmdln_args[i], std::atof(cmdln_args[i + 1]));
    params.type = std::atoi(cmdln_args[cmdln_count - 2]);
    params.SetInitialTemperature();
    if (params.type == 3 || params.type == 4 || params.type == 5) params.SetAnisotropicVariables();
    return params;
}

// ----------------------------------------

std::ostream& operator<<(std::ostream& out, SimulationParameters& params)
{
    Print(out, "#################################");
    Print(out, "# Parameters for exact solution #");
    Print(out, "#################################");
    Print(out, "tau_0    ", params.tau_0);
    Print(out, "tau_f    ", params.tau_f);
    Print(out, "Lambda_0 ", params.Lambda_0);
    Print(out, "xi_0     ", params.xi_0);
    Print(out, "alpha_0  ", params.alpha_0);
    Print(out, "mass     ", params.mass);
    Print(out, "C        ", params.C);
    Print(out, "steps    ", params.steps);
    Print(out, "step_size", params.step_size);
    Print(out, "\n");
    Print(out, "##################################");
    Print(out, "# Parameters for hyrdo evolution #");
    Print(out, "##################################");
    Print(out, "e0      ", params.e0);
    Print(out, "pl0     ", params.pl0);
    Print(out, "pt0     ", params.pt0);
    Print(out, "T0      ", params.T0);
    return out;
}

// ----------------------------------------

bool SimulationParameters::operator==(const SimulationParameters& other)
{
    bool is_match = (tau_0 == other.tau_0) && (Lambda_0 == other.Lambda_0) && (xi_0 == other.xi_0) && (alpha_0 == other.alpha_0)
                    && (tau_f == other.tau_f) && (mass == other.mass) && (C == other.C);
    return is_match;
}

bool SimulationParameters::operator!=(const SimulationParameters& other)
{
    return !(operator==(other));
}

void SimulationParameters::SetParameter(const char* name, double value)
{
    std::string var_name(name);
    if (var_name.compare("tau_0") == 0) tau_0 = value;
    else if (var_name.compare("e0") == 0) e0 = value;
    else if (var_name.compare("pt0") == 0) pt0 = value;
    else if (var_name.compare("pl0") == 0) pl0 = value;
    else if (var_name.compare("Lambda_0") == 0) Lambda_0 = value;
    else if (var_name.compare("xi_0") == 0) xi_0 = value;
    else if (var_name.compare("alpha_0") == 0) alpha_0 = value;
    else if (var_name.compare("tau_f") == 0) tau_f = value;
    else if (var_name.compare("mass") == 0) mass = value;
    else if (var_name.compare("C") == 0) C = value;
    else if (var_name.compare("pl0") == 0) pl0 = value;
    else if (var_name.compare("pt0") == 0) pt0 = value;

    if (var_name.compare("e0") == 0 || var_name.compare("mass") == 0) SetInitialTemperature();
    step_size = std::min(tau_0 / 20, 5.0 * C / T0);
    steps     = std::ceil((tau_f - tau_0) / step_size);
}

// ----------------------------------------

void SimulationParameters::SetParameters(double _tau_0, double _e0, double _pt0, double _pl0, double _tau_f, double _mass, double _C)
{
    tau_0 = _tau_0;
    e0    = _e0;
    pt0   = _pt0;
    pl0   = _pl0;
    tau_f = _tau_f;
    mass  = _mass;
    C     = _C;

    step_size = tau_0 / 20;
    steps     = std::ceil((tau_f - tau_0) / step_size);

    SetInitialTemperature();
    SetAnisotropicVariables();
}

// ----------------------------------------

void SimulationParameters::SetInitialConditions()
{
    hydro::AltAnisoHydroEvolution mvah;

    vec X = { alpha_0, Lambda_0, xi_0 };
    e0    = mvah.IntegralJ(2, 0, 0, 0, mass, X) / X(0);
    pt0   = mvah.IntegralJ(2, 0, 1, 0, mass, X) / X(0);
    pl0   = mvah.IntegralJ(2, 2, 0, 0, mass, X) / X(0);
    SetInitialTemperature();
}

// ----------------------------------------

void SimulationParameters::SetInitialTemperature()
{
    hydro::AltAnisoHydroEvolution mvah;
    T0 = mvah.InvertEnergyDensity(e0, mass);
}

void SimulationParameters::SetAnisotropicVariables()
{
    double x = std::log10(pt0 / pl0);
    vec    X = { 1.0, T0, 2.0 * std::pow(10.0, x) };
    FindAnisoVariables(e0, pt0, pl0, mass, X);
    alpha_0  = X(0);
    Lambda_0 = X(1);
    xi_0     = X(2);
}
