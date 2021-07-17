//
// Author: Kevin Ingles

#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/GlobalConstants.hpp"

#include <fstream>
#include <sstream>
#include <cmath>

SimulationParameters::SimulationParameters(const char* filename)
{
    std::fstream fin(filename, std::fstream::in);
    if (!fin.is_open())
    {
        Print_Error(std::cerr, "Failed to open file: ", filename);
        exit(-2);
    } // end if (!fin.is_open())

    const char hash = '#';
    const char endline = '\0';
    std::string line;
    std::stringstream buffer;
    std::string var_name;
    while (!fin.eof())
    {
        std::getline(fin, line);
        if (line[0] == hash || line[0] == endline) continue;
        else
        {
            buffer = std::stringstream(line);
            // Note: assumes tab or space separation
            buffer >> var_name;
            if (var_name.compare("tau_0") == 0)             buffer >> tau_0;
            else if (var_name.compare("Lambda_0") == 0)     buffer >> Lambda_0;
            else if (var_name.compare("xi_0") == 0)         buffer >> xi_0;
            else if (var_name.compare("alpha_0") == 0)      buffer >> alpha_0;
            else if (var_name.compare("ul") == 0)           buffer >> ul;
            else if (var_name.compare("ll") == 0)           buffer >> ll;
            else if (var_name.compare("mass") == 0)         buffer >> mass;
            else if (var_name.compare("eta_s") == 0)        buffer >> eta_s;
            else if (var_name.compare("steps") == 0)        buffer >> steps;
            else if (var_name.compare("pl0") == 0)          buffer >> pl0;
            else if (var_name.compare("pt0") == 0)          buffer >> pt0;
        } // end else
    } // end while(!fin.eof())
    step_size = tau_0 / 20;
    steps     = std::ceil((ul - ll) / step_size);

    SetInitialTemperature();
} // end SimulationParameters::SimulationParameters(...)
// -----------------------------------------

SimulationParameters::~SimulationParameters()
{
}

std::ostream& operator<<(std::ostream& out, SimulationParameters& params)
{
    Print(out, "#################################");
    Print(out, "# Parameters for exact solution #");
    Print(out, "#################################");
    Print(out, "tau_0    ", params.tau_0);
    Print(out, "Lambda_0 ", params.Lambda_0);
    Print(out, "xi_0     ", params.xi_0);
    Print(out, "alpha_0  ", params.alpha_0);
    Print(out, "ul       ", params.ul);
    Print(out, "ll       ", params.ll);
    Print(out, "mass     ", params.mass);
    Print(out, "eta_s    ", params.eta_s);
    Print(out, "steps    ", params.steps);
    Print(out, "step_size", params.step_size);
    Print(out, "\n");
    Print(out, "##################################");
    Print(out, "# Parameters for hyrdo evolution #");
    Print(out, "##################################");
    Print(out, "pl0     ", params.pl0);
    Print(out, "pt0     ", params.pt0);
    Print(out, "T0      ", params.T0);
    return out;
}
// ----------------------------------------

void SimulationParameters::SetParameter(const char* name, double value)
{
    std::string var_name(name);
    if (var_name.compare("tau_0") == 0)             tau_0    = value;
    else if (var_name.compare("Lambda_0") == 0)     Lambda_0 = value;
    else if (var_name.compare("xi_0") == 0)         xi_0     = value;
    else if (var_name.compare("alpha_0") == 0)      alpha_0  = value;
    else if (var_name.compare("ul") == 0)           ul       = value;
    else if (var_name.compare("ll") == 0)           ll       = value;
    else if (var_name.compare("mass") == 0)         mass     = value;
    else if (var_name.compare("eta_s") == 0)        eta_s    = value;
    else if (var_name.compare("pl0") == 0)          pl0      = value;
    else if (var_name.compare("pt0") == 0)          pt0      = value;


    if (var_name.compare("tau_0") == 0 || var_name.compare("ul") == 0 || var_name.compare("ll") == 0 || var_name.compare("steps") == 0)
        {
            step_size = tau_0 / 20;
            steps     = std::ceil((ul - ll) / step_size);
        }
    if (var_name.compare("Lambda_0") == 0 || var_name.compare("xi_0") == 0 || var_name.compare("alpha_0") == 0 || var_name.compare("mass") == 0)
        SetInitialTemperature();
}
// ----------------------------------------

void SimulationParameters::SetParameters(double _tau_0, double _Lambda_0, double _xi_0, double _alpah_0, double _ul, double _mass, double _eta_s)
{
    tau_0    = _tau_0;
    Lambda_0 = _Lambda_0;
    xi_0     = _xi_0;
    alpha_0  = alpha_0;
    ll       = _tau_0;
    ul       = _ul;
    mass     = _mass;
    eta_s    = _eta_s;
    
    step_size = tau_0 / 20;
    steps     = std::ceil((ll - ul) / step_size);

    SetInitialTemperature();
}

void SimulationParameters::SetInitialTemperature(void)
{
    // TO DO: add function that calculates the initial termperature
    // Functions used to calculate initial temperature
    auto H2 = [this](double y, double z)
    {
        if (std::fabs(y - 1.0) < 1e-15) y = 1 - 1e-15;

        if (std::fabs(y) < 1.0)
        {
            double x = std::sqrt((1.0 - y * y) / (y * y + z * z));
            return y * (std::sqrt(y * y + z * z) + (1.0 + z * z) / std::sqrt(1.0 - y * y) * std::atan(x));
        }
        else if (std::fabs(y) > 1.0)
        {
            double x = std::sqrt((y * y - 1.0) / (y * y + z * z));
            return y * (std::sqrt(y * y + z * z) + (1.0 + z * z) / std::sqrt(y * y - 1.0) * std::atanh(x));
        }
        else return 0.0;
    };

    auto H2Tilde = [this, H2](double y, double z)
    {
        return GausQuad([this, H2](double u, double y, double z)
        {
            return u * u * u * H2(y, z / u) * std::exp(-std::sqrt(u * u + z * z));
        }, 0, inf, 1e-5, 1, y, z);
    };

    auto diff = [this, H2Tilde](double T)
    {
        double z = mass / T;
        
        double Eeq;
        if (z == 0) Eeq = 3.0 * T * T * T * T / (PI * PI);
        else Eeq =  (3.0 * T * T * T * T) / (PI * PI) * ( z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);

        double Ean = std::pow(Lambda_0, 4.0) / (4.0 * PI * PI * alpha_0) * H2Tilde(1.0 / std::sqrt(1 + xi_0), mass / Lambda_0);

        return Eeq - Ean;
    };

    auto AnistoTropicToEquilibriumTermperature = [this, diff]()
    {
        double Thigh = 2.0 / 0.197;
        double Tlow  = 0.001 / 0.197;
        double copy  = Tlow;
        double err   = inf;

        double Tmid;
        while (err > 1e-15)
        {
            Tmid = (Thigh + Tlow) / 2.0;
            if (std::fabs(diff(Tmid)) < 1e-15) break;
            if (diff(Tmid) * diff(Tlow) <= 0)
                Thigh = Tmid;
            else 
                Tlow = Tmid;
            err = std::fabs(copy - Tmid);
            copy = Tmid;
        }

        return Tmid;
    };

    T0 = AnistoTropicToEquilibriumTermperature();
}