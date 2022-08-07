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
// File: HydroTheories.hpp
// Description: Defines the interface which implement the various hydrodynamic models
//              These inlcude: Chapman-Enskog, DNMR, VAH and or modification of VAH
//              called modified VAH

#ifndef HYDRO_THEORIES_HPP
#define HYDRO_THEORIES_HPP

#include "config.hpp"
#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Integration.hpp"
#include "Parameters.hpp"

#include <armadillo>
#include <vector>

using vec = arma::vec;
using mat = arma::mat;

using SP = SimulationParameters;

namespace hydro {
	enum class theory { CE = 0, DNMR };

	struct ViscousHydroEvolution {
		// Constructors
		ViscousHydroEvolution()	 = default;
		~ViscousHydroEvolution() = default;

		// Setup and run numerical evolution
		void RunHydroSimulation(const char* file_path, SP& params, theory theo);

		// Need to invert enery density to get temperature. This is done by taking advantage of
		// the Landau matching condition, i.e. the energy denisty in the comoving frame is
		// the same as the equilibrium energy density
		// TO DO: switch second argument to just `doublew mass`
		double EquilibriumEnergyDensity(double temp, SP& params);
		double InvertEnergyDensity(double e, SP& params);

		struct TransportCoefficients {
			double tau_pi;
			double beta_pi;
			double tau_Pi;
			double beta_Pi;
			double delta_pipi;
			double delta_PiPi;
			double lambda_piPi;
			double lambda_Pipi;
			double tau_pipi;
		};

		TransportCoefficients CalculateTransportCoefficients(double e, double pi, double Pi, SP& params, theory theo);

		// Evolution equations
		double dedt(double e, double p, double pi, double Pi, double tau);
		double dpidt(double pi, double Pi, double tau, TransportCoefficients& tc);
		double dPidt(double pi, double Pi, double tau, TransportCoefficients& tc);

		// Dynamical variables for RK4
		double e1, e2, e3, e4;
		double p1, p2, p3, p4;
		double pi1, pi2, pi3, pi4;
		double Pi1, Pi2, Pi3, Pi4;
		double de1, de2, de3, de4;
		double dpi1, dpi2, dpi3, dpi4;
		double dPi1, dPi2, dPi3, dPi4;
	};

	struct AnisoHydroEvolution {
		// Constructors
		AnisoHydroEvolution()  = default;
		~AnisoHydroEvolution() = default;

		// Setup and run numerical evolution
		void RunHydroSimulation(const char* file_path, SP& params);

		// Need to invert enery density to get temperature. This is done by taking advantage of
		// the Landau matching condition, i.e. the energy denisty in the comoving frame is
		// the same as the equilibrium energy density
		// TO DO: switch second argument to just `doublew mass`
		double EquilibriumEnergyDensity(double temp, SP& params);
		double InvertEnergyDensity(double e, SP& params);

		struct TransportCoefficients {
			double tau_pi;
			double tau_Pi;
			double zetaBar_zT;
			double zetaBar_zL;
		};

		TransportCoefficients CalculateTransportCoefficients(double e, double p, double pt, double pl, SP& params);
		// Functions used to calcualte the transport coefficients
		double InvertShearToXi(double e, double p, double pi);
		double R200(double xi);
		double R201(double xi);
		double R220(double xi);
		double R221(double xi);
		double R240(double xi);

		// Evolution equations
		double dedt(double e, double pl, double tau);
		double dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc);
		double dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc);

		// Dynamic variables for RK4: allocate here to make sure CPU has to constantly allocate new memory
		double e1, e2, e3, e4;
		double p1, p2, p3, p4;
		double pt1, pt2, pt3, pt4;
		double pl1, pl2, pl3, pl4;
		double de1, de2, de3, de4;
		double dpt1, dpt2, dpt3, dpt4;
		double dpl1, dpl2, dpl3, dpl4;

		// Simulation information
		double T0;	  // Starting temperature in fm^{-1}
	};

	struct AltAnisoHydroEvolution {
		// Constructors
		AltAnisoHydroEvolution()  = default;
		~AltAnisoHydroEvolution() = default;

		// Setup and run numerical evolution
		void RunHydroSimulation(const char* file_path, SP& params);

		// Need to invert enery density to get temperature. This is done by taking advantage of
		// the Landau matching condition, i.e. the energy denisty in the comoving frame is
		// the same as the equilibrium energy density
		double EquilibriumEnergyDensity(double temp, double mass);
		double InvertEnergyDensity(double e, double mass);

		struct TransportCoefficients {
			double tau_pi;
			double tau_Pi;
			double zetaBar_zT;
			double zetaBar_zL;
		};

		TransportCoefficients CalculateTransportCoefficients(double T, double pt, double pl, vec& X, SP& params);
		// Calculate Jacobian matrix to switch between hydro fields and anisotropic variables
		mat ComputeJacobian(double mass, const vec& X);
		// Functions used to calcualte the transport coefficients
		double IntegralJ(int n, int q, int r, int s, double mass, const vec& X);

		// Evolution equations
		double dedt(double e, double pl, double tau);
		double dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc);
		double dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc);

		// Dynamic variables for RK4: allocate here to make sure CPU has to constantly allocate new memory
		// TO DO: Correct operations of class
		double e1, e2, e3, e4;
		double p1, p2, p3, p4;
		double pt1, pt2, pt3, pt4;
		double pl1, pl2, pl3, pl4;
		double xi1, xi2, xi3, xi4;
		double dxi1, dxi2, dxi3, dxi4;
		double alpha1, alpha2, alpha3, alpha4;
		double dalpha1, dalpha2, dalpha3, dalpha4;
		double Lambda1, Lambda2, Lambda3, Lambda4;
		double dLambda1, dLambda2, dLambda3, dLambda4;

		vec X1, X2, X3, X4;
		vec psi1, psi2, psi3, psi4;
		vec qt1, qt2, qt3, qt4;

		// Simulation information
		double T0;	  // Starting temperature in fm^{-1}
	};
}	 // namespace hydro

#endif
