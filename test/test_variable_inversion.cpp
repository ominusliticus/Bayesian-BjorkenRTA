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
// File: test_variable_inversion.cpp
// Description: The purpose is to evaluate whether the numerical inversion of the hydrodynamic
//              fields to the anisotropic distribution variables works satisfactorily

#include "config.hpp"
#include "Errors.hpp"
#include "ExactSolution.hpp"
#include "GlobalConstants.hpp"
#include "HydroTheories.hpp"
#include "Integration.hpp"
#include "InvertObservables.hpp"
#include "Parameters.hpp"

#include <armadillo>
#include <cassert>

using SP  = SimulationParameters;
using vec = arma::vec;

int main()
{
	// SP params;
	// params.SetParameters(0.1, 1.64720, -0.9, 12.1, 0.654869, 0.200 / 0.197, 5.0 / (4.0 * PI));
	double mass	  = 0.2 / 0.197;
	double alpha  = 0.655;
	double Lambda = 0.5 / 0.197;
	double xi	  = -0.90;

	vec X	   = { alpha, Lambda, xi };
	vec X_soln = { alpha, Lambda, xi };

	hydro::AltAnisoHydroEvolution mvah;
	// The anisotropic intergrals that we care abour are
	// energy density: I(2, 0, 0, 0)
	// transverse pressure: I(2, 0, 1 , 0)
	// longitudinal pressure: I(2, 2, 0, 0)
	double e  = mvah.IntegralJ(2, 0, 0, 0, mass, X) / alpha;
	double pt = mvah.IntegralJ(2, 0, 1, 0, mass, X) / alpha;
	double pl = mvah.IntegralJ(2, 2, 0, 0, mass, X) / alpha;

	vec soln = { 1.0, 1.0, 1.0 };
	FindAnisoVariables(e, pt, pl, mass, soln);

	double tol = 1e-5;
	if (X_soln(0) - soln(0) < tol && X_soln(1) - soln(1) < tol && X_soln(2) - soln(2) < tol)
		Print(std::cout, "test_inversion: \033[1;32mPASSES!\033[0m");
	else Print(std::cout, "test_inversion: \033[1;31mFAILED!\033[0m");
	return 0;
}
