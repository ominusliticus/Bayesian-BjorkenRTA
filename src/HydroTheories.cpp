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
// File: HydroTheories.cpp
// Description: This file implements the various hydrodynamic models.
//              These include: Chapman-Enskog, DNMR, VAH and our own twist
//              on VAH called modified VAH

#include "HydroTheories.hpp"
#include "Integration.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

#if __APPLE__
namespace std {
	static double cyl_bessel_k(int n, double x)
	{
		return GausQuad(
			[n, x](double t) -> double
			{
				double val = std::exp(-x * std::cosh(t));
				return val > 0 ? val * std::cosh(static_cast<double>(n) * t) : 0;
			},
			0,
			inf,
			1e-8,
			8);
	}
}	 // namespace std
#endif

// TODO: Make easily parallelizable for python

namespace hydro {
	const double tol	   = 1e-15;
	const int	 max_depth = 1;

	// utility fuction for quick exponentiation
	double pow(double base, double exp)
	{
		if (base < 0) return std::exp(exp * std::log(-base)) * std::cos(exp * PI);
		else return std::exp(exp * std::log(base));
		;
	}

	// -------------------------------------

	double DoubleFactorial(int k)
	{
		if (k <= 1) return 1.0;
		double result = (double)k * DoubleFactorial(k - 2);
		return result;
	}

	// -------------------------------------

	double Gamma(double z)
	{
		return GausQuad(
			[](double x, double z)
			{
				return std::exp(-x) * pow(x, z - 1);
			},
			0,
			inf,
			tol,
			max_depth,
			z);
	}

	///////////////////////////////////
	// Viscous struct implementation //
	///////////////////////////////////
	void ViscousHydroEvolution::RunHydroSimulation(const char* file_path, SP& params, theory theo)
	{
		double t0 = params.tau_0;
		double dt = params.step_size;

		// for setprecision of t output
		int decimal = -(int)std::log10(dt);

		// Opening output files
		double				  m	   = params.mass;	 // Note that the mass in already in units fm^{-1}
		std::filesystem::path file = file_path;
		std::fstream		  e_plot, shear_plot, bulk_plot;
		switch (theo)
		{
			case theory::CE :
				// Print(std::cout, "Calculting viscous hydro in Chapman-Enskog approximation");
				e_plot	   = std::fstream(file / fmt::format("ce_e_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				shear_plot = std::fstream(file / fmt::format("ce_shear_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				bulk_plot  = std::fstream(file / fmt::format("ce_bulk_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				break;
			case theory::DNMR :
				// Print(std::cout, "Calculting viscous hydro in 14-moment approximation");
				e_plot	   = std::fstream(file / fmt::format("dnmr_e_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				shear_plot = std::fstream(file / fmt::format("dnmr_shear_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				bulk_plot  = std::fstream(file / fmt::format("dnmr_bulk_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
				break;
		}
		if (!e_plot.is_open() && !shear_plot.is_open() && !bulk_plot.is_open())
		{
			Print_Error(std::cerr, "ViscousHydroEvolution::RunHydroSimulation: Failed to open output files.");
			switch (theo)
			{
				case theory::CE :
					Print_Error(std::cerr, "Pleae make sure the folder ./output/CE_hydro/ exists.");
					break;
				case theory::DNMR :
					Print_Error(std::cerr, "Pleae make sure the folder ./output/DNMR_hydro/ exists.");
					break;
			}
			exit(-3333);
		}
		else
		{
			e_plot << std::fixed << std::setprecision(decimal + 10);
			shear_plot << std::fixed << std::setprecision(decimal + 10);
			bulk_plot << std::fixed << std::setprecision(decimal + 10);
		}

		// Initialize simulation
		double T0 = params.T0;	  // Note that the temperature is already in units fm^{-1}
		double z0 = m / T0;
		double e0;	  // Equilibrium energy density
		if (z0 == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
		else
			e0 = 3.0 * pow(T0, 4.0) / (PI * PI)
				 * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

		// Thermal pressure necessary for calculating bulk pressure and inverting pi to xi
		auto ThermalPressure = [this](double e, SP& params) -> double
		{
			double T = InvertEnergyDensity(e, params);
			double z = params.mass / T;
			if (z == 0) return pow(T, 4.0) / (PI * PI);
			else return z * z * pow(T, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_k(2, z);
		};

		// Note: all dynamic variables are declared as struct memebrs variables
		e1	= e0;
		pi1 = 2.0 * (params.pt0 - params.pl0) / 3.0;
		Pi1 = (params.pl0 + 2.0 * params.pt0) / 3.0 - ThermalPressure(e0, params);

		// Begin simulation
		TransportCoefficients tc;
		double				  t;
		for (int n = 0; n < params.steps; n++)
		{
			t = t0 + n * dt;

			p1 = ThermalPressure(e1, params);
			Print(e_plot, t, e1, p1);
			Print(shear_plot, t, pi1);
			Print(bulk_plot, t, Pi1);

			// Invert energy density to compute thermal pressure

			// RK4 with updating anisotropic variables
			// Note all dynamic variables are declared as member variables
			// fmt::print("e1 = {}, pi1 = {}, Pi1 = {}\n", e1, pi1, Pi1);

			// First order
			tc	 = CalculateTransportCoefficients(e1, pi1, Pi1, params, theo);
			de1	 = dt * dedt(e1, p1, pi1, Pi1, t);
			dpi1 = dt * dpidt(pi1, Pi1, t, tc);
			dPi1 = dt * dPidt(pi1, Pi1, t, tc);

			e2	= e1 + de1 / 2.0;
			pi2 = pi1 + dpi1 / 2.0;
			Pi2 = Pi1 + dPi1 / 2.0;

			// Second order
			p2	 = ThermalPressure(e2, params);
			tc	 = CalculateTransportCoefficients(e2, pi2, Pi2, params, theo);
			de2	 = dt * dedt(e2, p2, pi2, Pi2, t + dt / 2.0);
			dpi2 = dt * dpidt(pi2, Pi2, t + dt / 2.0, tc);
			dPi2 = dt * dPidt(pi2, Pi2, t + dt / 2.0, tc);

			e3	= e1 + de2 / 2.0;
			pi3 = pi1 + dpi2 / 2.0;
			Pi3 = Pi1 + dPi2 / 2.0;

			// Third order
			p3	 = ThermalPressure(e3, params);
			tc	 = CalculateTransportCoefficients(e3, pi3, Pi3, params, theo);
			de3	 = dt * dedt(e3, p3, pi3, Pi3, t + dt / 2.0);
			dpi3 = dt * dpidt(pi3, Pi3, t + dt / 2.0, tc);
			dPi3 = dt * dPidt(pi3, Pi3, t + dt / 2.0, tc);

			e4	= e1 + de3;
			pi4 = pi1 + dpi3;
			Pi4 = Pi1 + dPi3;

			// Fourth order
			p4	 = ThermalPressure(e4, params);
			tc	 = CalculateTransportCoefficients(e4, pi4, Pi4, params, theo);
			de4	 = dt * dedt(e4, p4, pi4, Pi4, t + dt);
			dpi4 = dt * dpidt(pi4, Pi4, t + dt, tc);
			dPi4 = dt * dPidt(pi4, Pi4, t + dt, tc);

			e1 += (de1 + 2.0 * de2 + 2.0 * de3 + de4) / 6.0;
			pi1 += (dpi1 + 2.0 * dpi2 + 2.0 * dpi3 + dpi4) / 6.0;
			Pi1 += (dPi1 + 2.0 * dPi2 + 2.0 * dPi3 + dPi4) / 6.0;
		}	 // End simulation loop

		e_plot.close();
		shear_plot.close();
		bulk_plot.close();
	}

	// -------------------------------------

	double ViscousHydroEvolution::EquilibriumEnergyDensity(double temp, SP& params)
	{
		double z = params.mass / temp;
		if (z == 0) return 3.0 * pow(temp, 4.0) / (PI * PI);
		else
			return 3.0 * pow(temp, 4.0) / (PI * PI)
				   * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
	}

	// -------------------------------------

	double ViscousHydroEvolution::InvertEnergyDensity(double e, SP& params)
	{
		double x1, x2, mid;
		double T_min = .001 / .197;
		double T_max = 2.0 / .197;
		x1			 = T_min;
		x2			 = T_max;

		double copy(0.0), prec = 1.e-6;
		int	   n	  = 0;
		int	   flag_1 = 0;
		do
		{
			mid			 = (x1 + x2) / 2.0;
			double e_mid = EquilibriumEnergyDensity(mid, params);
			double e1	 = EquilibriumEnergyDensity(x1, params);

			if (abs(e_mid - e) < prec) break;

			if ((e_mid - e) * (e1 - e) <= 0.0) x2 = mid;
			else x1 = mid;

			n++;
			if (n == 1) copy = mid;

			if (n > 4)
			{
				if (abs(copy - mid) < prec) flag_1 = 1;
				copy = mid;
			}
		} while (flag_1 != 1 && n <= 2000);

		return mid;
	}

	// -------------------------------------

	ViscousHydroEvolution::TransportCoefficients
	ViscousHydroEvolution::CalculateTransportCoefficients(double e, double pi, double Pi, SP& params, theory theo)
	{
		// invert energy density to temperature
		double T	= InvertEnergyDensity(e, params);
		double m	= params.mass;
		double z	= m / T;
		double beta = 1.0 / T;

		switch (theo)
		{
			case theory::CE :
			{
				// Thermodynamic integrals, c.f. Eqs. (45) - (48) in arXiv:1407.7231
				double I3_63, I1_42, I3_42, I2_31, I0_31, I0_30;
				if (z == 0)
				{
					I3_63 = -4.0 * pow(T, 5.0) / (35.0 * PI * PI);
					I1_42 = 4.0 * pow(T, 5.0) / (5.0 * PI * PI);
					I3_42 = pow(T, 3.0) / (15.0 * PI * PI);
					I2_31 = -pow(T, 3.0) / (3.0 * PI * PI);
					I0_31 = -4.0 * e * T / 3.0;
					I0_30 = 4.0 * e * T;
				}
				else
				{
					double K5 = std::cyl_bessel_k(5, z);
					double K3 = std::cyl_bessel_k(3, z);
					double K2 = std::cyl_bessel_k(2, z);
					double K1 = std::cyl_bessel_k(1, z);

					double Ki3 = GausQuad(
						[](double th, double z)
						{
							return std::exp(-z * std::cosh(th)) * pow(std::cosh(th), -3);
						},
						0,
						inf,
						tol,
						max_depth,
						z);
					double Ki1 = GausQuad(
						[](double th, double z)
						{
							return std::exp(-z * std::cosh(th)) * pow(std::cosh(th), -1);
						},
						0,
						inf,
						tol,
						max_depth,
						z);

					I3_63
						= -pow(T * z, 5.0) / (210 * PI * PI) * ((K5 - 11.0 * K3 + 58.0 * K1) / 16.0 - 4.0 * Ki1 + Ki3);
					I1_42 = pow(T * z, 5.0) / (30.0 * PI * PI) * ((K5 - 7.0 * K3 + 22.0 * K1) / 16.0 - Ki1);
					I3_42 = pow(T * z, 3.0) / (30.0 * PI * PI) * ((K3 - 9.0 * K1) * 0.25 + 3.0 * Ki1 - Ki3);
					I2_31 = -pow(T * z, 3.0) / (6.0 * PI * PI) * ((K3 - 5.0 * K1) * 0.25 + Ki1);
					I0_31 = -(e + z * z * pow(T, 4.0) / (2.0 * PI * PI) * K2) * T;
					I0_30 = (3.0 * e + (3.0 + z * z) * z * z * pow(T, 4.0) / (2.0 * PI * PI) * K2) * T;
				}

				// sound speed sqrd
				double cs2 = -I0_31 / I0_30;

				// calculatiung transport coefficients
				// Eqs. (25) - (26) in arXiv:1407.7231
				double beta_pi = beta * I1_42;
				double beta_Pi = 5.0 * beta_pi / 3.0 + beta * I0_31 * cs2;
				// double s = - beta * beta * I0_31;                             // thermal entropy density

				// TO DO: should relaxation time always be the same in Bjorken flow?
				double tau_pi = 5.0 * params.C / T;
				double tau_Pi = tau_pi;

				// Eqs. (35) - (40) arXiv:1407:7231
				double chi = beta * ((1.0 - 3.0 * cs2) * (I1_42 + I0_31) - m * m * (I3_42 + I2_31)) / beta_Pi;
				if (z == 0) chi = -9.0 / 5.0;
				double delta_PiPi  = -5.0 * chi / 9.0 - cs2;
				double lambda_Pipi = beta * (7.0 * I3_63 + 2.0 * I1_42) / (3.0 * beta_pi) - cs2;
				double tau_pipi	   = 2.0 + 4.0 * beta * I3_63 / beta_pi;
				double delta_pipi  = 5.0 / 3.0 + 7.0 * beta * I3_63 / (3.0 * beta_pi);
				double lambda_piPi = -2.0 * chi / 3.0;

				// double check1 = std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2);
				// double check2 = std::fabs(lambda_Pipi - delta_pipi + 1 + cs2);
				// double check3 = std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0);

				TransportCoefficients tc{ tau_pi,	  beta_pi,	   tau_Pi,		beta_Pi, delta_pipi,
										  delta_PiPi, lambda_piPi, lambda_Pipi, tau_pipi };
				return tc;
				// double local_tol = 100 * tol;
				// if (check1 < local_tol && check2 < local_tol && check3 < local_tol) return tc;
				// else
				// {
				//     Print_Error(std::cerr, "ViscousHydroEvolution::CalculateTransportCoefficients: transport
				//     coefficients did not satisfy relations."); Print_Error(std::cerr,
				//     fmt::format("std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2)      = {}", check1));
				//     Print_Error(std::cerr, fmt::format("std::fabs(lambda_Pipi - delta_pipi + 1 + cs2)              =
				//     {}", check2)); Print_Error(std::cerr, fmt::format("std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi
				//     - 1.0) / 7.0) = {}", check3)); exit(-5555);
				// }
				break;
			}

			case theory::DNMR :	   // Reference Appendix E in arXiv:1803.01810
			{
				// See arXiv:1403.0962 Eq. (48) for this definition, where z = cosh(th)
				auto IntegralI = [](int n, int q, double z, double T)
				{
					if (z == 0) { return pow(T, n + 2) * Gamma(n + 2) / (2.0 * PI * PI * DoubleFactorial(2 * q + 1)); }
					else
					{
						return GausQuad(
							[](double x, double z, double T, int n, int q)
							{
								return pow(T * z, n + 2) * pow(x, n - 2 * q) * pow(x * x - 1.0, (2 * q + 1) * 0.5)
									   * std::exp(-z * x) / (2.0 * PI * PI * DoubleFactorial(2 * q + 1));
							},
							1,
							inf,
							tol,
							max_depth,
							z,
							T,
							n,
							q);
					}
				};

				double I00 = IntegralI(0, 0, z, T);
				double I01 = IntegralI(0, 1, z, T);
				double I22 = IntegralI(2, 2, z, T);
				double I30 = IntegralI(3, 0, z, T);
				double I31 = IntegralI(3, 1, z, T);
				double I32 = IntegralI(3, 2, z, T);
				double I40 = IntegralI(4, 0, z, T);
				double I41 = IntegralI(4, 1, z, T);
				double I42 = IntegralI(4, 2, z, T);

				// sound speed sqrd
				double cs2	   = I31 / I30;
				double cBar_e  = -I41 / (5.0 * I40 * I42 / 3.0 - I41 * I41);
				double cBar_Pi = I40 / (5.0 * I40 * I42 / 3.0 - I41 * I41);
				if (z == 0)
				{
					cBar_e	= 0;
					cBar_Pi = 0;
				}
				double cBar_pi = 0.5 / I42;

				double beta_pi = beta * I32;
				double beta_Pi = 5.0 * beta_pi / 3.0 - beta * I31 * cs2;
				// double s = beta * beta * I31;                             // thermal entropy density

				double tau_pi = 5.0 * params.C / T;
				double tau_Pi = tau_pi;

				double delta_PiPi  = 1.0 - cs2 - pow(m, 4.0) * (cBar_e * I00 + cBar_Pi * I01) / 9.0;
				double lambda_Pipi = (1.0 + cBar_pi * m * m * I22) / 3.0 - cs2;
				double tau_pipi	   = (10.0 + 4.0 * cBar_pi * m * m * I22) / 7.0;
				double delta_pipi  = (4.0 + cBar_pi * m * m * I22) / 3.0;
				;
				double lambda_piPi = 6.0 / 5.0 - 2.0 * pow(m, 4.0) * (cBar_e * I00 + cBar_Pi * I01) / 15;

				// double check1 = std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2);
				// double check2 = std::fabs(lambda_Pipi - delta_pipi + 1 + cs2);
				// double check3 = std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0);

				TransportCoefficients tc{ tau_pi,	  beta_pi,	   tau_Pi,		beta_Pi, delta_pipi,
										  delta_PiPi, lambda_piPi, lambda_Pipi, tau_pipi };
				return tc;
				// double local_tol = 100 * tol;
				// if (check1 < local_tol && check2 < local_tol && check3 < local_tol) return tc;
				// else
				// {
				//     Print_Error(std::cerr, "ViscousHydroEvolution::CalculateTransportCoefficients: transport
				//     coefficients did not satisfy relations."); Print_Error(std::cerr,
				//     fmt::format("std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2)      = {}", check1));
				//     Print_Error(std::cerr, fmt::format("std::fabs(lambda_Pipi - delta_pipi + 1 + cs2)              =
				//     {}", check2)); Print_Error(std::cerr, fmt::format("std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi
				//     - 1.0) / 7.0) = {}", check3)); exit(-5555);
				// }
				break;
			}
		}	 // End switch(theo)

		return {};
	}

	// -------------------------------------

	double ViscousHydroEvolution::dedt(double e, double p, double pi, double Pi, double tau)
	{
		return -(e + p + Pi - pi) / tau;
	}

	// -------------------------------------

	double ViscousHydroEvolution::dpidt(double pi, double Pi, double tau, TransportCoefficients& tc)
	{
		return -pi / tc.tau_pi + 4.0 * tc.beta_pi / (3.0 * tau) - (tc.tau_pipi / 3.0 + tc.delta_pipi) * pi / tau
			   + 2.0 * tc.lambda_piPi * Pi / (3.0 * tau);
	}

	// -------------------------------------

	double ViscousHydroEvolution::dPidt(double pi, double Pi, double tau, TransportCoefficients& tc)
	{
		return -Pi / tc.tau_Pi - tc.beta_Pi / tau - tc.delta_PiPi * Pi / tau + tc.lambda_Pipi * pi / tau;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////
	// Anisotropic struct implementation //
	///////////////////////////////////////
	void AnisoHydroEvolution::RunHydroSimulation(const char* file_path, SP& params)
	{
		// Print(std::cout, "Calculating anistropic hydrodynamic evolution");
		double t0 = params.tau_0;
		double dt = params.step_size;

		// for setprecision of t output
		int decimal = -(int)std::log10(dt);

		// Opening output files
		double				  m	   = params.mass;	 // Note that the mass in already in units fm^{-1}
		std::filesystem::path file = file_path;
		std::fstream		  e_plot(file / fmt::format("vah_e_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		std::fstream		  bulk_plot(file / fmt::format("vah_bulk_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		std::fstream		  shear_plot(file / fmt::format("vah_shear_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		if (!e_plot.is_open() && !bulk_plot.is_open() && !shear_plot.is_open())
		{
			Print_Error(std::cerr, "AnisoHydroEvolution::RunHydroSimulation: Failed to open output files.");
			Print_Error(std::cerr, "Pleae make sure the folder ./output/aniso_hydro/ exists.");
			exit(-3333);
		}
		else
		{
			e_plot << std::fixed << std::setprecision(decimal + 10);
			bulk_plot << std::fixed << std::setprecision(decimal + 10);
			shear_plot << std::fixed << std::setprecision(decimal + 10);
		}

		// Initialize simulation
		T0		  = params.T0;	  // Note that the temperature is already in units fm^{-1}
		double z0 = m / T0;
		double e0;	  // Equilibrium energy density
		if (z0 == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
		else
			e0 = 3.0 * pow(T0, 4.0) / (PI * PI)
				 * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

		// Thermal pressure necessary for calculating bulk pressure and inverting pi to xi
		auto ThermalPressure = [this](double e, SP& params) -> double
		{
			double T = InvertEnergyDensity(e, params);
			double z = params.mass / T;
			if (z == 0) return pow(T, 4.0) / (PI * PI);
			else return z * z * pow(T, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_k(2, z);
		};

		// Note: all dynamic variables are declared as struct memebrs variables
		e1 = e0;

		pt1 = params.pt0;
		pl1 = params.pl0;
		p1	= ThermalPressure(e1, params);

		// Begin simulation
		TransportCoefficients tc = CalculateTransportCoefficients(e1, p1, pt1, pl1, params);
		double				  t;
		double				  xi = params.xi_0;
		for (int n = 0; n < params.steps; n++)
		{
			t		  = t0 + n * dt;
			p1		  = ThermalPressure(e1, params);
			double pi = 2.0 / 3.0 * (pt1 - pl1);
			double Pi = (2.0 * pt1 + pl1) / 3.0 - p1;
			xi		  = InvertShearToXi(e1, p1, pi);

			Print(e_plot, t, e1, p1, pl1, pt1, xi);
			Print(bulk_plot, t, Pi, tc.zetaBar_zT);
			Print(shear_plot, t, pi, tc.zetaBar_zL);

			// RK4 with updating anisotropic variables
			// Note all dynamic variables are declared as member variables

			// First order
			tc	 = CalculateTransportCoefficients(e1, p1, pt1, pl1, params);
			de1	 = dt * dedt(e1, pl1, t);
			dpt1 = dt * dptdt(p1, pt1, pl1, t, tc);
			dpl1 = dt * dpldt(p1, pt1, pl1, t, tc);

			e2	= e1 + de1 / 2.0;
			pt2 = pt1 + dpt1 / 2.0;
			pl2 = pl1 + dpl1 / 2.0;

			// Second order
			p2	 = ThermalPressure(e2, params);
			pi	 = 2.0 / 3.0 * (pt2 - pl2);
			Pi	 = (pl2 + 2.0 * pt2) / 3.0 - p2;
			tc	 = CalculateTransportCoefficients(e2, p2, pt2, pl2, params);
			de2	 = dt * dedt(e2, pl2, t + dt / 2.0);
			dpt2 = dt * dptdt(p2, pt2, pl2, t + dt / 2.0, tc);
			dpl2 = dt * dpldt(p2, pt2, pl2, t + dt / 2.0, tc);

			e3	= e1 + de2 / 2.0;
			pt3 = pt1 + dpt2 / 2.0;
			pl3 = pl1 + dpl2 / 2.0;

			// Third order
			p3	 = ThermalPressure(e3, params);
			pi	 = 2.0 / 3.0 * (pt3 - pl3);
			Pi	 = (pl3 + 2.0 * pt3) / 3.0 - p3;
			tc	 = CalculateTransportCoefficients(e3, p3, pt3, pl3, params);
			de3	 = dt * dedt(e3, pl3, t + dt / 2.0);
			dpt3 = dt * dptdt(p3, pt3, pl3, t + dt / 2.0, tc);
			dpl3 = dt * dpldt(p3, pt3, pl3, t + dt / 2.0, tc);

			e4	= e1 + de3;
			pt4 = pt1 + dpt3;
			pl4 = pl1 + dpl3;

			// Fourth order
			p4	 = ThermalPressure(e4, params);
			pi	 = 2.0 / 3.0 * (pt4 - pl4);
			Pi	 = (pl4 + 2.0 * pt4) / 3.0 - p4;
			tc	 = CalculateTransportCoefficients(e4, p4, pt4, pl4, params);
			de4	 = dt * dedt(e4, pl4, t + dt);
			dpt4 = dt * dptdt(p4, pt4, pl4, t + dt, tc);
			dpl4 = dt * dpldt(p4, pt4, pl4, t + dt, tc);

			e1 += (de1 + 2.0 * de2 + 2.0 * de3 + de4) / 6.0;
			pt1 += (dpt1 + 2.0 * dpt2 + 2.0 * dpt3 + dpt4) / 6.0;
			pl1 += (dpl1 + 2.0 * dpl2 + 2.0 * dpl3 + dpl4) / 6.0;

		}	 // End simulation loop

		e_plot.close();
		bulk_plot.close();
		shear_plot.close();
	}

	// -------------------------------------

	double AnisoHydroEvolution::EquilibriumEnergyDensity(double temp, SP& params)
	{
		double z = params.mass / temp;
		if (z == 0) return 3.0 * pow(temp, 4.0) / (PI * PI);
		else
			return 3.0 * pow(temp, 4.0) / (PI * PI)
				   * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
	}

	// -------------------------------------

	double AnisoHydroEvolution::InvertEnergyDensity(double e, SP& params)
	{
		double x1, x2, mid;
		double T_min = .001 / .197;
		double T_max = 2.0 / .197;
		x1			 = T_min;
		x2			 = T_max;

		double copy(0.0), prec = 1.e-10;
		int	   n	  = 0;
		int	   flag_1 = 0;
		do
		{
			mid			 = (x1 + x2) / 2.0;
			double e_mid = EquilibriumEnergyDensity(mid, params);
			double e1	 = EquilibriumEnergyDensity(x1, params);

			if (abs(e_mid - e) < prec) break;

			if ((e_mid - e) * (e1 - e) <= 0.0) x2 = mid;
			else x1 = mid;

			n++;
			if (n == 1) copy = mid;

			if (n > 4)
			{
				if (abs(copy - mid) < prec) flag_1 = 1;
				copy = mid;
			}
		} while (flag_1 != 1 && n <= 2000);

		return mid;
	}

	// -------------------------------------

	AnisoHydroEvolution::TransportCoefficients
	AnisoHydroEvolution::CalculateTransportCoefficients(double e, double p, double pt, double pl, SP& params)
	{
		double T = InvertEnergyDensity(e, params);

		// Coefficients for relaxation times
		// TO DO: should the relaxation times always be equal in Bjorken flow?
		double tau_pi = 5.0 * params.C / T;
		double tau_Pi = tau_pi;

		// Calculate shear pressure
		double pi = 2.0 / 3.0 * (pt - pl);
		double xi = InvertShearToXi(e, p, pi);

		// Calculate transport coefficients
		double zetaBar_zL = 1.0 * e * R240(xi) / R200(xi) - 3.0 * pl;
		double zetaBar_zT = 1.0 * (e / 2.0) * R221(xi) / R200(xi) - pt;
		// if (params.mass == 0) zetaBar_zL = -(e + pl + 2.0 * zetaBar_zT);

		TransportCoefficients tc{ tau_pi, tau_Pi, zetaBar_zT, zetaBar_zL };
		return tc;
	}

	// -------------------------------------

	double AnisoHydroEvolution::InvertShearToXi(double e, double p, double pi)
	{
		if (pi == 0) return 0.0;

		double err		 = inf;
		double local_tol = 1e-10;
		double xi1		 = -0.5;
		double xi2		 = 0.5;
		double xin		 = 0.0;

		double piBar = pi / (e + p);
		double alpha = 1.0e-1;	  // search speed;

		// function we want to find the root of
		auto func = [this](double piBar, double xi) -> double
		{
			return piBar + 0.25 * (3.0 * R220(xi) / R200(xi) - 1.0);
		};

		while (err > local_tol)
		{
			xin = xi2 - alpha * func(piBar, xi2) * (xi2 - xi1) / (func(piBar, xi2) - func(piBar, xi1));

			// Check for zero anisotropy case
			if (xin == 0) return 0;

			// calculate error and update variables
			err = std::fabs(xin - xi2) / std::fabs(xin);
			xi1 = xi2;
			xi2 = xin;
		}

		return xin;
	}

	// -------------------------------------

	double AnisoHydroEvolution::R200(double xi)
	{
		if (xi == 0) return 1.0;
		else if (xi < 0) return 0.5 * (1.0 / (1.0 + xi) + std::atanh(std::sqrt(-xi)) / std::sqrt(-xi));
		else return 0.5 * (1.0 / (1.0 + xi) + std::atan(std::sqrt(xi)) / std::sqrt(xi));
	}

	// -------------------------------------

	double AnisoHydroEvolution::R220(double xi)
	{
		if (xi == 0) return 0.3333333;
		else if (xi < 0) return 0.5 * (-1.0 / (1.0 + xi) + std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / xi;
		else return 0.5 * (-1.0 / (1.0 + xi) + std::atan(std::sqrt(xi)) / std::sqrt(xi)) / xi;
	}

	// -------------------------------------

	double AnisoHydroEvolution::R201(double xi)
	{
		if (xi == 0) return 0.5;
		else if (xi < 0) return 0.5 * (1.0 - (1.0 - xi) * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / xi;
		else return 0.5 * (1.0 - (1.0 - xi) * std::atan(std::sqrt(xi)) / std::sqrt(xi)) / xi;
	}

	double AnisoHydroEvolution::R221(double xi)
	{
		if (xi == 0) return 4.0 / 30.0;
		else if (xi < 0) return 0.5 * (-3.0 + (3.0 + xi) * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / pow(xi, 2.0);
		else return 0.5 * (-3.0 + (3.0 + xi) * std::atan(std::sqrt(xi)) / std::sqrt(xi)) / pow(xi, 2.0);
	}

	// -------------------------------------

	double AnisoHydroEvolution::R240(double xi)
	{
		if (xi == 0) return 0.2;
		else if (xi < 0)
			return 0.5 * ((3.0 + 2.0 * xi) / (1.0 + xi) - 3.0 * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi))
				   / pow(xi, 2.0);
		else
			return 0.5 * ((3.0 + 2.0 * xi) / (1.0 + xi) - 3.0 * std::atan(std::sqrt(xi)) / std::sqrt(xi))
				   / pow(xi, 2.0);
	}

	// -------------------------------------

	double AnisoHydroEvolution::dedt(double e, double pl, double tau)
	{
		return -(e + pl) / tau;
	}

	// -------------------------------------

	double AnisoHydroEvolution::dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
	{
		double tau_pi	  = tc.tau_pi;
		double tau_Pi	  = tc.tau_Pi;
		double zetaBar_zT = tc.zetaBar_zT;
		double pbar		  = (pl + 2.0 * pt) / 3.0;
		return -(pbar - p) / tau_Pi + (pl - pt) / (3.0 * tau_pi) + zetaBar_zT / tau;
	}

	// -------------------------------------

	double AnisoHydroEvolution::dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
	{
		double tau_pi	  = tc.tau_pi;
		double tau_Pi	  = tc.tau_Pi;
		double zetaBar_zL = tc.zetaBar_zL;
		double pbar		  = (pl + 2.0 * pt) / 3.0;
		return -(pbar - p) / tau_Pi - (pl - pt) / (1.5 * tau_pi) + zetaBar_zL / tau;
	}

	// -----------------------------------------

	///////////////////////////////////////////
	// Alt Anisotropic struct implementation //
	//////////////.////////////////////////////

	// Thermal pressure necessary for calculating bulk pressure and inverting pi to xi
	double ThermalPressure(double T, double mass)
	{
		double z = mass / T;
		if (z == 0) return pow(T, 4.0) / (PI * PI);
		else return z * z * pow(T, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_k(2, z);
	}

	// -----------------------------------------

	// Function does in intermittent RK step and checks if xi > -1
	// If not, it calls itself with a subdivision of the current interval (t,t+dt)
	// to improve the "convergence" (not sure what to call it)
	void AltAnisoHydroEvolution::RK4Update(vec&					  X_current,
										   vec&					  X_update,
										   vec&					  dX,
										   double				  t,
										   double				  dt,
										   double				  T,
										   size_t				  steps,
										   TransportCoefficients& tc,
										   const SP&			  params)
	{
		double m = params.mass;

		// RK4 with updating anisotropic variables
		// Note all dynamic variables are declared as member variables
		X1		  = X_current;
		vec dummy = { 0.0, 0.0, 0.0 };
		dX		  = dummy;

		for (size_t n = 0; n < steps; ++n)
		{
			// First order
			// Calculate Jacobian matrix for (E, PT, PL) -> (alpha, Lambda, xi)
			mat M = ComputeJacobian(m, X1);

			// compute transport coefficients to calculate evolution of (E,PT,PL) and store in vector
			tc	 = CalculateTransportCoefficients(T, pt1, pl1, X1, params);
			psi1 = { dedt(e1, pl1, t), dptdt(p1, pt1, pl1, t, tc), dpldt(p1, pt1, pl1, t, tc) };

			// Convert evolution vector to (alpha, Lambda, xi) coordinates
			qt1 = M.i() * psi1;
			// Print(std::cout, M);
			// Print(std::cout, M.i());
			// Print(std::cout, psi1);
			// Print(std::cout, qt1);
			// Print(std::cout, "--------------------------");

			// Calculate update step
			dX1 = dt * qt1;
			X2	= X1 + 0.5 * dX1;

			// Second order
			if (X2(2) < -0.999)												// Check if xi < -1.0
				RK4Update(X1, X2, dX1, t, dt / 10.0, T, 10, tc, params);	// Poorly placed but best I could come up
																			// with to implement recursion

			e2	= IntegralJ(2, 0, 0, 0, m, X2) / X2(0);
			pt2 = IntegralJ(2, 0, 1, 0, m, X2) / X2(0);
			pl2 = IntegralJ(2, 2, 0, 0, m, X2) / X2(0);
			// Print(std::cout, e2, pl2, pt2);
			// Print(std::cout, alpha2, Lambda2, xi2);
			// Print(std::cout, "--------------------------");

			T	 = InvertEnergyDensity(e2, m);
			p2	 = ThermalPressure(T, m);
			M	 = ComputeJacobian(m, X2);
			tc	 = CalculateTransportCoefficients(T, pt2, pl2, X2, params);
			psi2 = { dedt(e2, pl2, t + dt / 2.0),
					 dptdt(p2, pt2, pl2, t + dt / 2.0, tc),
					 dpldt(p2, pt2, pl2, t + dt / 2.0, tc) };
			qt2	 = M.i() * psi2;

			dX2 = dt * qt2;
			X3	= X1 + 0.5 * dX2;

			// Third order
			if (X3(2) < -0.999) RK4Update(X2, X3, dX2, t, dt / 20.0, T, 10, tc, params);

			e3	= IntegralJ(2, 0, 0, 0, m, X3) / X3(0);
			pt3 = IntegralJ(2, 0, 1, 0, m, X3) / X3(0);
			pl3 = IntegralJ(2, 2, 0, 0, m, X3) / X3(0);
			// Print(std::cout, e3, pl3, pt3);
			// Print(std::cout, alpha3, Lambda3, xi3);
			// Print(std::cout, "--------------------------");

			T	 = InvertEnergyDensity(e3, m);
			p3	 = ThermalPressure(T, m);
			M	 = ComputeJacobian(m, X3);
			tc	 = CalculateTransportCoefficients(T, pt3, pl3, X3, params);
			psi3 = { dedt(e3, pl3, t + dt / 2.0),
					 dptdt(p3, pt3, pl3, t + dt / 2.0, tc),
					 dpldt(p3, pt3, pl3, t + dt / 2.0, tc) };
			qt3	 = M.i() * psi3;

			dX3 = dt * qt3;
			X4	= X1 + dX3;

			// Fourth order
			if (X4(2) < -0.999) RK4Update(X3, X4, dX3, t, dt / 20.0, T, 10, tc, params);

			e4	= IntegralJ(2, 0, 0, 0, m, X4) / X4(0);
			pt4 = IntegralJ(2, 0, 1, 0, m, X4) / X4(0);
			pl4 = IntegralJ(2, 2, 0, 0, m, X4) / X4(0);
			// Print(std::cout, e4, pl4, pt4);
			// Print(std::cout, alpha4, Lambda4, xi4);
			// Print(std::cout, "--------------------------");
			// Print(std::cout);

			T	 = InvertEnergyDensity(e4, m);
			p4	 = ThermalPressure(T, m);
			M	 = ComputeJacobian(m, X4);
			tc	 = CalculateTransportCoefficients(T, pt4, pl4, X4, params);
			psi4 = { dedt(e4, pl4, t + dt), dptdt(p4, pt4, pl4, t + dt, tc), dpldt(p4, pt4, pl4, t + dt, tc) };
			qt4	 = M.i() * psi4;

			dX4 = dt * qt4;
			if (X1(2) + dX4(2) < -0.999) RK4Update(X4, dummy, dX4, t, dt / 10.0, T, 10, tc, params);

			dX += (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4) / 6.0;
			X1 = X1 + (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4) / 6.0;

			// update first step values
			e1	= IntegralJ(2, 0, 0, 0, m, X1) / X1(0);
			pt1 = IntegralJ(2, 0, 1, 0, m, X1) / X1(0);
			pl1 = IntegralJ(2, 2, 0, 0, m, X1) / X1(0);
			T	= InvertEnergyDensity(e1, m);
			p1	= ThermalPressure(T, m);
		}

		X_update = X1;
	}

	// -----------------------------------------

	void AltAnisoHydroEvolution::RunHydroSimulation(const char* file_path, const SP& params)
	{
		// Print(std::cout, "Calculating alternative anistropic hydrodynamic evolution");
		double t0 = params.tau_0;
		double dt = params.step_size;

		// for setprecision of t output
		int decimal = -(int)std::log10(dt);

		// Opening output files
		double				  m	   = params.mass;	 // Note that the mass in already in units fm^{-1}
		std::filesystem::path file = file_path;
		std::fstream		  e_plot(file / fmt::format("mvah_e_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		std::fstream		  bulk_plot(file / fmt::format("mvah_bulk_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		std::fstream		  shear_plot(file / fmt::format("mvah_shear_m={:.3f}GeV.dat", 0.197 * m), std::ios::out);
		if (!e_plot.is_open() && !bulk_plot.is_open() && !shear_plot.is_open())
		{
			Print_Error(std::cerr, "AnisoHydroEvolution::RunHydroSimulation: Failed to open output files.");
			Print_Error(std::cerr, "Pleae make sure the folder ./output/aniso_hydro/ exists.");
			exit(-3333);
		}
		else
		{
			e_plot << std::fixed << std::setprecision(decimal + 10);
			bulk_plot << std::fixed << std::setprecision(decimal + 10);
			shear_plot << std::fixed << std::setprecision(decimal + 10);
		}

		// Initialize simulation
		T0		  = params.T0;	  // Note that the temperature is already in units fm^{-1}
		double z0 = m / T0;
		double e0;	  // Equilibrium energy density
		if (z0 == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
		else
			e0 = 3.0 * pow(T0, 4.0) / (PI * PI)
				 * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

		// Note: all dynamic variables are declared as struct memebrs variables
		// Initializing simulation variables
		e1 = e0;
		p1 = ThermalPressure(T0, m);

		double alpha1  = params.alpha_0;
		double Lambda1 = params.Lambda_0;
		double xi1	   = params.xi_0;

		X1 = { alpha1, Lambda1, xi1 };

		pt1 = params.pt0;
		pl1 = params.pl0;

		// Begin simulation
		TransportCoefficients tc = CalculateTransportCoefficients(T0, pt1, pl1, X1, params);
		double				  t;
		double				  T = T0;
		mat					  M;
		vec					  X{ X1 }, X_old{ X1 }, dX{ X1 };
		double				  e{ e1 }, pt{ pt1 }, pl{ pl1 }, p{ p1 }, xi{ xi1 };
		for (int n = 0; n < params.steps; n++)
		{
			t = t0 + n * dt;

			double pi = 2.0 * (pt - pl) / 3.0;
			double Pi = (2.0 * pt + pl) / 3.0 - p;
			Print(e_plot, t, e, p, pt, pl, xi);
			Print(bulk_plot, t, Pi, tc.zetaBar_zT);
			Print(shear_plot, t, pi, tc.zetaBar_zL);

			RK4Update(X_old, X, dX, t, dt, T, 1, tc, params);
			e  = IntegralJ(2, 0, 0, 0, m, X) / X(0);
			pt = IntegralJ(2, 0, 1, 0, m, X) / X(0);
			pl = IntegralJ(2, 2, 0, 0, m, X) / X(0);
			T  = InvertEnergyDensity(e, m);
			p  = ThermalPressure(T, m);

			X_old = X;

		}	 // End simulation loop

		e_plot.close();
		bulk_plot.close();
		shear_plot.close();
	}

	// -------------------------------------

	double AltAnisoHydroEvolution::EquilibriumEnergyDensity(double temp, double mass)
	{
		double z = mass / temp;
		if (z == 0) return 3.0 * pow(temp, 4.0) / (PI * PI);
		else
			return 3.0 * pow(temp, 4.0) / (PI * PI)
				   * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
	}

	// -------------------------------------

	double AltAnisoHydroEvolution::InvertEnergyDensity(double e, double mass)
	{
		double x1, x2, mid;
		double T_min = .001 / .197;
		double T_max = 2.0 / .197;
		x1			 = T_min;
		x2			 = T_max;

		double copy(0.0), prec = 1.e-10;
		int	   n	  = 0;
		int	   flag_1 = 0;
		do
		{
			mid			 = (x1 + x2) / 2.0;
			double e_mid = EquilibriumEnergyDensity(mid, mass);
			double e1	 = EquilibriumEnergyDensity(x1, mass);

			if (abs(e_mid - e) < prec) break;

			if ((e_mid - e) * (e1 - e) <= 0.0) x2 = mid;
			else x1 = mid;

			n++;
			if (n == 1) copy = mid;

			if (n > 4)
			{
				if (abs(copy - mid) < prec) flag_1 = 1;
				copy = mid;
			}
		} while (flag_1 != 1 && n <= 2000);

		return mid;
	}

	// -------------------------------------
	mat AltAnisoHydroEvolution::ComputeJacobian(double m, const vec& X)
	{
		double a = X(0);
		double L = X(1);
		mat	   M = { { -IntegralJ(2, 0, 0, 0, m, X) / (a * a),
					   IntegralJ(2, 0, 0, 1, m, X) / (a * L * L),
					   -IntegralJ(4, 2, 0, -1, m, X) / (2.0 * a * L) },
					 { -IntegralJ(2, 0, 1, 0, m, X) / (a * a),
					   IntegralJ(2, 0, 1, 1, m, X) / (a * L * L),
					   -IntegralJ(4, 2, 1, -1, m, X) / (2.0 * a * L) },
					 { -IntegralJ(2, 2, 0, 0, m, X) / (a * a),
					   IntegralJ(2, 2, 0, 1, m, X) / (a * L * L),
					   -IntegralJ(4, 4, 0, -1, m, X) / (2.0 * a * L) } };
		return M;
	};

	// -------------------------------------

	double AltAnisoHydroEvolution::IntegralJ(int n, int r, int q, int s, double mass, const vec& X)
	{
		double Lambda  = X(1);
		double xi	   = X(2);
		double alpha_L = 1.0 / std::sqrt(1.0 + xi);
		double alpha_T = 1.0;
		double m_bar   = mass / Lambda;
		double norm	   = std::pow(alpha_T, 2 * q + 2) * std::pow(alpha_L, r + 1) * std::pow(Lambda, n + s + 2)
					  / (4.0 * PI * PI * DoubleFactorial(2.0 * q));

		auto Rnrq = [=](double p_bar)
		{
			double w  = std::sqrt(alpha_L * alpha_L + std::pow(m_bar / p_bar, 2.0));
			double w3 = w * w * w;
			double z  = (alpha_T * alpha_T - alpha_L * alpha_L) / (w * w);
			double z2{ z * z }, z3{ z2 * z }, z4{ z3 * z }, z5{ z4 * z };
			double t = 0;
			if (z == 0) t = 0;
			else if (z < 0) t = std::atanh(std::sqrt(-z)) / std::sqrt(-z);
			else t = std::atan(std::sqrt(z)) / std::sqrt(z);

			if (std::fabs(z) < 0.1)
			{
				if (n == 2 && r == 0 && q == 0)
				{
					return 2.0 * w * (1.0 + z / 3.0 - z2 / 15.0 + z3 / 35.0 - z4 / 63.0 + z5 / 99.0);
				}
				else if (n == 2 && r == 0 && q == 1)
				{
					return 4.0 / w * (1.0 / 3.0 - z / 15.0 + z2 / 35.0 - z3 / 63.0 + z4 / 99.0 - z5 / 143.0);
				}
				else if (n == 2 && r == 2 && q == 0)
				{
					return 2.0 / w * (1.0 / 3.0 - z / 15.0 + z2 / 35.0 - z3 / 63.0 + z4 / 99.0 - z5 / 143.0);
				}
				else if (n == 2 && r == 2 && q == 1)
				{
					return 4.0 / w3
						   * (1.0 / 15.0 - 2.0 * z / 35.0 + z2 / 21.0 - 4.0 * z3 / 99.0 + 5.0 * z4 / 143.0
							  - 2.0 * z5 / 65.0);
				}
				else if (n == 2 && r == 4 && q == 0)
				{
					return 2.0 / w3
						   * (1.0 / 5.0 - 3.0 * z / 35.0 + z2 / 21.0 - z3 / 33.0 + 3.0 * z4 / 143.0 - z5 / 65.0);
				}
				else if (n == 4 && r == 2 && q == 0)
				{
					return 2.0 * w * (1.0 / 3.0 + z / 15.0 - z2 / 105.0 + z3 / 315.0 - z4 / 693.0 + z5 / 1287.0);
				}
				else if (n == 4 && r == 2 && q == 1)
				{
					return 4.0 / w
						   * (1.0 / 15.0 - 2.0 * z / 105.0 + z2 / 105.0 - 4.0 * z3 / 693.0 + 5.0 * z4 / 1287.0
							  - 2.0 * z5 / 715.0);
				}
				else if (n == 4 && r == 4 && q == 0)
				{
					return 2.0 / w * (1.0 / 5.0 - z / 35.0 + z2 / 105.0 - z3 / 231.0 + z4 / 429.0 - z5 / 715.0);
				}
				else assert("Unsupported choice");
			}
			else
			{
				if (n == 2 && r == 0 && q == 0) return w * (1.0 + (1.0 + z) * t);
				else if (n == 2 && r == 0 && q == 1) return (1.0 + (z - 1.0) * t) / (z * w);
				else if (n == 2 && r == 2 && q == 0) return (-1.0 + (1.0 + z) * t) / (z * w);
				else if (n == 2 && r == 2 && q == 1) return (-3.0 + (3.0 + z) * t) / (z * z * w * w * w);
				else if (n == 2 && r == 4 && q == 0) return (3.0 + 2.0 * z - 3.0 * (1.0 + z) * t) / (z * z * w * w * w);
				else if (n == 4 && r == 2 && q == 0)
					return (w * (-1.0 + z + (1.0 + z) * (1.0 + z) * t)) / (4.0 * z);	// I calculated by hand
				else if (n == 4 && r == 2 && q == 1) return (3.0 + z + (1.0 + z) * (z - 3.0) * t) / (4.0 * z * z * w);
				else if (n == 4 && r == 4 && q == 0)
					return (-(3.0 + 5.0 * z) + 3.0 * (1.0 + z) * (1.0 + z) * t)
						   / (4.0 * z * z * w);	   // I calculated by hand
				else assert("Unsupported choice");
			}
			return -inf;
		};

		auto integrand = [=](double p_bar)
		{
			return std::pow(p_bar, n + s + 1) * std::pow(1.0 + std::pow(m_bar / p_bar, 2.0), (double)s / 2.0)
				   * Rnrq(p_bar) * std::exp(-std::sqrt(p_bar * p_bar + m_bar * m_bar));
		};

		double result = norm * GausQuad(integrand, 0, inf, tol, max_depth);
		return result;
	}

	AltAnisoHydroEvolution::TransportCoefficients
	AltAnisoHydroEvolution::CalculateTransportCoefficients(double T, double pt, double pl, vec& X, const SP& params)
	{
		// Coefficients for relaxation times
		// TO DO: should the relaxation times always be equal in Bjorken flow?
		double tau_pi = 5.0 * params.C / T;
		double tau_Pi = tau_pi;

		// Calculate transport coefficients
		double zetaBar_zL = IntegralJ(2, 4, 0, 0, params.mass, X) / X(0) - 3.0 * pl;
		double zetaBar_zT = IntegralJ(2, 2, 1, 0, params.mass, X) / X(0) - pt;
		// if (params.mass == 0) zetaBar_zL = -(e + pl + 2.0 * zetaBar_zT);

		TransportCoefficients tc{ tau_pi, tau_Pi, zetaBar_zT, zetaBar_zL };
		return tc;
	}

	// -------------------------------------

	double AltAnisoHydroEvolution::dedt(double e, double pl, double tau)
	{
		return -(e + pl) / tau;
	}

	// -------------------------------------

	double AltAnisoHydroEvolution::dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
	{
		double tau_pi	  = tc.tau_pi;
		double tau_Pi	  = tc.tau_Pi;
		double zetaBar_zT = tc.zetaBar_zT;
		double pbar		  = (pl + 2.0 * pt) / 3.0;
		return -(pbar - p) / tau_Pi + (pl - pt) / (3.0 * tau_pi) + zetaBar_zT / tau;
	}

	// -------------------------------------

	double AltAnisoHydroEvolution::dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
	{
		double tau_pi	  = tc.tau_pi;
		double tau_Pi	  = tc.tau_Pi;
		double zetaBar_zL = tc.zetaBar_zL;
		double pbar		  = (pl + 2.0 * pt) / 3.0;
		return -(pbar - p) / tau_Pi - (pl - pt) / (1.5 * tau_pi) + zetaBar_zL / tau;
	}

	// -----------------------------------------

}	 // namespace hydro

