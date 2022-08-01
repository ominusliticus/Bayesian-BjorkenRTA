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
// File: main.cpp
// Description: Translation unit that contains the int main program and

#include "config.hpp"
#include "Errors.hpp"
#include "ExactSolution.hpp"
#include "GlobalConstants.hpp"
#include "HydroTheories.hpp"
#include "Integration.hpp"
#include "Parameters.hpp"

#include <chrono>

int main(int argc, char** argv)
{
	SimulationParameters params = std::move(SimulationParameters::ParseCmdLine(argc, argv));

	switch (params.type)
	{
		case 0 :	// Chapman-Enskog hydro
		{
			hydro::ViscousHydroEvolution hydro_ce;
			hydro_ce.RunHydroSimulation(argv[argc - 1], params, hydro::theory::CE);
			hydro_ce.~ViscousHydroEvolution();
			break;
		}
		case 1 :	// DNMR hydro
		{
			hydro::ViscousHydroEvolution hydro_dnmr;
			hydro_dnmr.RunHydroSimulation(argv[argc - 1], params, hydro::theory::DNMR);
			hydro_dnmr.~ViscousHydroEvolution();
			break;
		}
		case 2 :	// Anisotropic
		{
			hydro::AnisoHydroEvolution hydro_aniso;
			hydro_aniso.RunHydroSimulation(argv[argc - 1], params);
			hydro_aniso.~AnisoHydroEvolution();
			break;
		}
		case 3 :	// Alternative anisotropic formulation
		{
			hydro::AltAnisoHydroEvolution hydro_altaniso;
			hydro_altaniso.RunHydroSimulation(argv[argc - 1], params);
			hydro_altaniso.~AltAnisoHydroEvolution();
			break;
		}
		case 4 :	// Boltzmann RTA exact solution
		{
			exact::ExactSolution hydro_exact;
			hydro_exact.Run(params);
			hydro_exact.OutputMoments(argv[argc - 1], params);
			break;
		}
		default :
		{
			Print(std::cout, "Invalid variable SimulationParameter::type. Please fix.");
			exit(-1234);
		}
	}
	return 0;
}
