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
#include "GlobalConstants.hpp"
#include "HydroTheories.hpp"
#include "Parameters.hpp"

#include <cassert>
#include <cmath>

using SP  = SimulationParameters;
using vec = arma::vec;
static hydro::AltAnisoHydroEvolution evo;

/// hydro_fields is vector of the form (E, PT, PL)
/// aniso_vars is vector of the form (alpha, Lambda, xi)
vec&& ComputeF(const vec& hydro_fields, double mass, const vec& aniso_vars)
{
	double micro_energy_density = evo.IntegralJ(2, 0, 0, 0, mass, aniso_vars) / aniso_vars(0);
	double micro_trans_pressure = evo.IntegralJ(2, 0, 1, 0, mass, aniso_vars) / aniso_vars(0);
	double micro_long_pressure	= evo.IntegralJ(2, 2, 0, 0, mass, aniso_vars) / aniso_vars(0);

	return { micro_energy_density - hydro_fields(0),
			 micro_long_pressure - hydro_fields(1),
			 micro_trans_pressure - hydro_fields(2) };
}

/// hydro_fields is vector of the form (E, PT, PL)
/// aniso_vars is vector of the form (alpha, Lambda, xi)
double LineBackTrack(const vec& hydro_fields, const double& aniso_vars, const double& delta_ansio_vars, double mass)
{
	vec	   aniso_vars_update = aniso_var;
	vec	   F				 = ComputeF(hydro_fields(0), hydro_fields(1), hydro_fields(2), mass, aniso_vars_update);
	double mag_F			 = arma::norm(F, 2);
}

// ----------------------------------------

double FindAnisoVariables();

