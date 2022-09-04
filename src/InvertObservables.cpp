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
// File: InvertObservables.cpp
// Descripition: This file implements the multi-dimensional numerical inverter
//               which allows us to convert observables such as PL and PT, or
//               Pi, pi and P to the microscopic parameters needed by VAH,
//               modified VAH and the RTA solution.

#include "InvertObservables.hpp"
#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "HydroTheories.hpp"
#include "Parameters.hpp"

#include <cassert>
#include <cmath>

using SP  = SimulationParameters;
using vec = arma::vec;
static hydro::AltAnisoHydroEvolution evo;

/// hydro_fields is vector of the form (E, PT, PL)
/// aniso_vars is vector of the form (Log(alpha), Lambda, xi)
vec ComputeF(const vec& hydro_fields, double mass, const vec& aniso_vars)
{
    double micro_energy_density = evo.IntegralJ(2, 0, 0, 0, mass, aniso_vars) / aniso_vars(0);
    double micro_trans_pressure = evo.IntegralJ(2, 0, 1, 0, mass, aniso_vars) / aniso_vars(0);
    double micro_long_pressure  = evo.IntegralJ(2, 2, 0, 0, mass, aniso_vars) / aniso_vars(0);

    vec local = { micro_energy_density - hydro_fields(0),
                  micro_trans_pressure - hydro_fields(1),
                  micro_long_pressure - hydro_fields(2) };
    return local;
}

/// hydro_fields is vector of the form (E, PT, PL)
/// aniso_vars is vector of the form (Log(alpha), Lambda, xi)
/// Line backtracing algorithm taken from Numerical Recipes pgs. 478-489
double LineBackTrack(const vec& hydro_fields, const vec& aniso_vars, const vec& delta_aniso_vars, double mass)
{
    vec    aniso_vars_update = aniso_vars;
    vec    F                 = ComputeF(hydro_fields, mass, aniso_vars_update);
    double mag_F2            = 0.5 * std::pow(arma::norm(F, 2), 2.0);
    double mag_dX            = arma::norm(delta_aniso_vars, 2);

    double step_adj       = 1.;           ///< parameter returned by line brack-trace algo
    double alpha          = 1e-4;         ///< Descent rate
    double g0             = mag_F2;       ///< g(x) is aux function to help us minimize search
    double g0_prime       = -2.0 * g0;    ///< g'(x) evaluated at x_0
    double step_adj_root  = -g0_prime / (2.0 * (mag_F2 - g0 - g0_prime));    // Starting guess
    double step_adj_prev  = step_adj_root;
    double mag_F2_current = mag_F2;
    double mag_F2_prev    = mag_F2;
    for (int i = 0; i < 10; ++i)
    {
        if (step_adj * mag_dX <= tol_dX) return step_adj;                        // Check if converged
        else if (mag_F2 <= g0 + step_adj * alpha * g0_prime) return step_adj;    // Check if converging fast enough
        else
        {
            double a = (mag_F2_current - g0 - step_adj * g0_prime) / (step_adj * step_adj);
            a -= (mag_F2_prev - g0 - step_adj_prev * g0_prime) / (step_adj_prev * step_adj_prev);
            a /= (step_adj - step_adj_prev);
            double b = -step_adj_prev * (mag_F2_current - g0 - step_adj * g0_prime) / (step_adj * step_adj);
            b += step_adj * (mag_F2_prev - g0 - step_adj_prev * g0_prime) / (step_adj_prev * step_adj_prev);
            b /= (step_adj - step_adj_prev);

            if (a == 0) step_adj_root = -g0_prime / (2.0 * b);    // root if g(x) is quadratic
            else
            {
                double z = b * b - 3.0 * a * g0_prime;
                if (z < 0) step_adj_root = 0.5 * step_adj;
                else if (b <= 0) step_adj_root = (-b + std::sqrt(z)) / (3.0 * a);
                else step_adj_root = -g0_prime / (b + std::sqrt(z));
            }
            step_adj_root = std::fmin(step_adj_root, 0.5 * step_adj);
        }
        step_adj_prev     = step_adj;
        mag_F2_prev       = mag_F2_current;
        step_adj          = std::fmax(step_adj_root, 0.1 * step_adj);
        aniso_vars_update = aniso_vars + step_adj * delta_aniso_vars;    // Might want to insert alpha here
        F                 = ComputeF(hydro_fields, mass, aniso_vars_update);
        mag_F2_current    = 0.5 * std::pow(arma::norm(F, 2), 2.0);
    }
    return step_adj;
}

// ----------------------------------------

void FindAnisoVariables(double E, double PT, double PL, double mass, vec& aniso_vars)
{
    constexpr double step_max = 10.0;
    // The aniso variables are of the form (Log(alpha), Lambda, xi)
    vec       delta_aniso_vars = { 0.0, 0.0, 0.0 };
    const vec hydro_fields     = { E, PT, PL };
    vec       F                = ComputeF(hydro_fields, mass, aniso_vars);
    bool      converged        = false;
    // for (size_t n = 0; n < 10000; ++n)
    size_t n = 0;
    while (!converged)
    {
        mat J            = evo.ComputeJacobian(mass, aniso_vars);
        delta_aniso_vars = -J.i() * F;
        // rescale if difference is too large
        double mag_delta_aniso_vars = arma::norm(delta_aniso_vars, 2);
        if (mag_delta_aniso_vars > step_max)
        {
            for (auto& x : delta_aniso_vars)
                x *= step_max / mag_delta_aniso_vars;
            mag_delta_aniso_vars = step_max;
        }
        double step_adj = LineBackTrack(hydro_fields, aniso_vars, delta_aniso_vars, mass);
        // Update aniso variables
        aniso_vars = aniso_vars + step_adj * delta_aniso_vars;
        F          = ComputeF(hydro_fields, mass, aniso_vars);
        if (aniso_vars(0) < 0.0 || aniso_vars(1) < 0.0 || aniso_vars(2) < -1.0)
            Print(std::cout, "Variable inversion gave unphysical anisotropic parameters.");
        // Check for convergence
        if (step_adj * mag_delta_aniso_vars < tol_dX && arma::norm(F, 2) < tol_F)
        {
            converged = true;
            return;
        }
        ++n;
    }
    Print(std::cerr, "Failed to converge within steps");
    Print(std::cout, "Failed to converge: convergence should not fail. Try increasing N_max");
}

