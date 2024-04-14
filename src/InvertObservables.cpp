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
vec ComputeF(const vec& hydro_fields, double mass, const vec& aniso_vars, bool b_2dim)
{
    if (b_2dim)
    {
        double micro_energy_density = evo.IntegralJ(2, 0, 0, 0, mass, vec{ 1.0, aniso_vars(0), aniso_vars(1) });
        double micro_trans_pressure = evo.IntegralJ(2, 0, 1, 0, mass, vec{ 1.0, aniso_vars(0), aniso_vars(1) });

        vec local = { micro_energy_density - hydro_fields(0), micro_trans_pressure - hydro_fields(1) };
        return local;
    }
    else
    {
        double alpha                = aniso_vars(0);
        double micro_energy_density = evo.IntegralJ(2, 0, 0, 0, mass, aniso_vars) / alpha;
        double micro_trans_pressure = evo.IntegralJ(2, 0, 1, 0, mass, aniso_vars) / alpha;
        double micro_long_pressure  = evo.IntegralJ(2, 2, 0, 0, mass, aniso_vars) / alpha;

        vec local = { micro_energy_density - hydro_fields(0),
                      micro_trans_pressure - hydro_fields(1),
                      micro_long_pressure - hydro_fields(2) };
        return local;
    }
}

/// hydro_fields is vector of the form (E, PT, PL)
/// aniso_vars is vector of the form (Log(alpha), Lambda, xi)
/// Line backtracing algorithm taken from Numerical Recipes pgs. 478-489
static double LineBackTrack(const vec& hydro_fields, const vec& aniso_vars, const vec& delta_aniso_vars, double mass, bool b_2dim)
{
    vec    aniso_vars_update = aniso_vars;
    vec    F                 = ComputeF(hydro_fields, mass, aniso_vars_update, b_2dim);
    double mag_F2            = 0.5 * std::pow(arma::norm(F, 2), 2.0);
    double mag_dX            = arma::norm(delta_aniso_vars, 2);

    double step_adj       = 1.;                                              ///< parameter returned by line brack-trace algo
    double alpha          = 1e-4;                                            ///< Descent rate
    double g0             = mag_F2;                                          ///< g(x) is aux function to help us minimize search
    double g0_prime       = -2.0 * g0;                                       ///< g'(x) evaluated at x_0
    double step_adj_root  = -g0_prime / (2.0 * (mag_F2 - g0 - g0_prime));    // Starting guess
    double step_adj_prev  = step_adj_root;
    double mag_F2_current = mag_F2;
    double mag_F2_prev    = mag_F2;
    for (int i = 0; i < 10; ++i)
    {
        if (step_adj * mag_dX <= tol_dX) return step_adj;                                // Check if converged
        else if (mag_F2_current <= g0 + step_adj * alpha * g0_prime) return step_adj;    // Check if converging fast enough
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
        F                 = ComputeF(hydro_fields, mass, aniso_vars_update, b_2dim);
        mag_F2_current    = 0.5 * std::pow(arma::norm(F, 2), 2.0);
    }
    return step_adj;
}

// ----------------------------------------

void FindAnisoVariables(double E, double PT, double PL, double mass, vec& aniso_vars, bool b_2dim)
{
    constexpr double max_step_size = 10.0;

    auto iterate = [mass, b_2dim, max_step_size](const vec& hydro_fields, vec& aniso_vars, vec& delta_aniso_vars, vec& F)
    {
        bool   converged = false;
        size_t n         = 0;
        while (!converged)
        {
            // Print(std::cout, F);
            mat J = evo.ComputeJacobian(mass, b_2dim ? vec{ 1.0, aniso_vars(0), aniso_vars(1) } : aniso_vars);
            if (b_2dim)
            {
                auto Jprime      = mat{ { J(0, 1), J(0, 2) }, { J(1, 1), J(1, 2) } };
                delta_aniso_vars = -Jprime.i() * F;
            }
            else delta_aniso_vars = -J.i() * F;
            // rescale if difference is too large
            double mag_delta_aniso_vars = arma::norm(delta_aniso_vars, 2);
            if (mag_delta_aniso_vars > max_step_size)
            {
                for (auto& x : delta_aniso_vars)
                    x *= max_step_size / mag_delta_aniso_vars;
                mag_delta_aniso_vars = max_step_size;
            }
            double step_adj = LineBackTrack(hydro_fields, aniso_vars, delta_aniso_vars, mass, b_2dim);
            // Update aniso variables
            aniso_vars = aniso_vars + step_adj * delta_aniso_vars;
            F          = ComputeF(hydro_fields, mass, aniso_vars, b_2dim);
            // TODO: Check for unphysical values in the the cases where we have 2d and 3d inversion
            // if (aniso_vars(0) < 0.0 || aniso_vars(1) < 0.0 || aniso_vars(2) < -1.0)
            // {
            //     Print(std::cout, "Variable inversion gave unphysical anisotropic parameters.");
            //     aniso_vars(0) = 1.0;
            //     aniso_vars(1) = 0.0;
            //     aniso_vars(2) = 0.0;
            //     return;
            // }
            // if (n % 100 == 0)
            // {
            //     Print(std::cout, aniso_vars);
            //     Print(std::cout, hydro_fields);
            //     Print(std::cout, F);
            //     Print(std::cout, J);
            //     Print(std::cout, arma::norm(F, 2), tol_F);
            //     Print(std::cout, step_adj * mag_delta_aniso_vars, tol_dX);
            //     Print(std::cout, "------------------");
            // }

            // Check for convergence
            if (b_2dim && step_adj * mag_delta_aniso_vars < tol_dX)
            {
                converged = true;
                return;
            }
            if (step_adj * mag_delta_aniso_vars < tol_dX && arma::norm(F, 2) < tol_F)
            {
                converged = true;
                return;
            }
            ++n;
        }
        Print(std::cerr, "Failed to converge within steps");
    };

    // The aniso variables are of the form (Log(alpha), Lambda, xi)
    if (b_2dim)
    {
        vec       delta_aniso_vars = { 0.0, 0.0 };
        const vec hydro_fields     = { E, PT };
        vec       F                = ComputeF(hydro_fields, mass, aniso_vars, b_2dim);
        iterate(hydro_fields, aniso_vars, delta_aniso_vars, F);
    }
    else
    {
        vec       delta_aniso_vars = { 0.0, 0.0, 0.0 };
        const vec hydro_fields     = { E, PT, PL };
        vec       F                = ComputeF(hydro_fields, mass, aniso_vars, b_2dim);
        iterate(hydro_fields, aniso_vars, delta_aniso_vars, F);
    }
}
