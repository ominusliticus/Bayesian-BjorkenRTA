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
// File: Parameter.hpp
// Description: Header file that describes interface for argument parsing input streams
//              to generate a struct with parameters needed to run the various
//              hydrodynamic models

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "Errors.hpp"

#include <armadillo>
#include <string>
#include <vector>

using vec = arma::vec;

struct SimulationParameters {
	SimulationParameters() = default;
	SimulationParameters(const char* filename);
	~SimulationParameters();

	static SimulationParameters ParseCmdLine(int cmdln_count, char** cmdln_args);

	void SetParameter(const char* name, double value);
	void SetParameters(double _tau_0,
					   double _Lambda_0,
					   double _xi_0,
					   double _alpah_0,
					   double _tau_f,
					   double _mass,
					   double _eta_s);
	void SetInitialTemperature();
	void SetAnisotropicVariables();

	friend std::ostream& operator<<(std::ostream& out, SimulationParameters& params);

	bool operator==(const SimulationParameters& other);
	bool operator!=(const SimulationParameters& other);

	unsigned int type;	  // Which hydro simulation to compute

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
	double e0;
	double pl0;
	double pt0;
	double T0;

	std::string file_identifier;
};	  // end struct SimulationParameters

#endif
