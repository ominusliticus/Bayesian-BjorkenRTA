//
// Author: Kevin Ingles

#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/GlobalConstants.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cassert>


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
            else if (var_name.compare("tau_f") == 0)        buffer >> tau_f;
            else if (var_name.compare("mass") == 0)         buffer >> mass;
            else if (var_name.compare("C") == 0)            buffer >> C;
            else if (var_name.compare("steps") == 0)        buffer >> steps;
            else if (var_name.compare("TYPE") == 0)         buffer >> type;
            else if (var_name.compare("FILE") == 0)         buffer >> file_identifier;
        } // end else
    } // end while(!fin.eof())
    step_size = tau_0 / 20;
    steps     = std::ceil((tau_f - tau_0) / step_size);

    SetInitialTemperature();
    fin.close();
} // end SimulationParameters::SimulationParameters(...)
// -----------------------------------------

SimulationParameters::~SimulationParameters()
{
}

SimulationParameters SimulationParameters::ParseCmdLine(int cmdln_count, char** cmdln_args)
{
    if (cmdln_count == 0)
        return SimulationParameters("utils/params.txt");

    SimulationParameters params{};
    for (int i = 0; i < cmdln_count; i += 2)
        params.SetParameter(cmdln_args[i], std::atof(cmdln_args[i+1]));

    return params;
}
// ----------------------------------------

std::ostream& operator<<(std::ostream& out, SimulationParameters& params)
{
    Print(out, "#################################");
    Print(out, "# Parameters for exact solution #");
    Print(out, "#################################");
    Print(out, "tau_0    ", params.tau_0);
    Print(out, "Lambda_0 ", params.Lambda_0);
    Print(out, "xi_0     ", params.xi_0);
    Print(out, "alpha_0  ", params.alpha_0);
    Print(out, "tau_f    ", params.tau_f);
    Print(out, "mass     ", params.mass);
    Print(out, "C        ", params.C);
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

bool SimulationParameters::operator==(const SimulationParameters& other)
{
    bool is_match = (tau_0 == other.tau_0)
        && (Lambda_0 == other.Lambda_0)
        && (xi_0 == other.xi_0)
        && (alpha_0 == other.alpha_0)
        && (tau_f == other.tau_f)
        && (mass == other.mass)
        && (C == other.C);
    return is_match;
}

bool SimulationParameters::operator!=(const SimulationParameters& other)
{
    return !(operator==(other));
}

void SimulationParameters::SetParameter(const char* name, double value)
{
    std::string var_name(name);
    if (var_name.compare("tau_0") == 0)             tau_0    = value;
    else if (var_name.compare("Lambda_0") == 0)     Lambda_0 = value;
    else if (var_name.compare("xi_0") == 0)         xi_0     = value;
    else if (var_name.compare("alpha_0") == 0)      alpha_0  = value;
    else if (var_name.compare("tau_f") == 0)        tau_f    = value;
    else if (var_name.compare("mass") == 0)         mass     = value;
    else if (var_name.compare("C") == 0)            C        = value;
    else if (var_name.compare("pl0") == 0)          pl0      = value;
    else if (var_name.compare("pt0") == 0)          pt0      = value;


    if (var_name.compare("tau_0") == 0 || var_name.compare("tau_f") == 0 || var_name.compare("steps") == 0)
        {
            step_size = tau_0 / 20;
            steps     = std::ceil((tau_f - tau_0) / step_size);
        }
    if (var_name.compare("Lambda_0") == 0 || var_name.compare("xi_0") == 0 || var_name.compare("alpha_0") == 0 || var_name.compare("mass") == 0)
        SetInitialTemperature();
}
// ----------------------------------------

void SimulationParameters::SetParameters(double _tau_0, double _Lambda_0, double _xi_0, double _alpha_0, double _tau_f, double _mass, double _C)
{
    tau_0    = _tau_0;
    Lambda_0 = _Lambda_0;
    xi_0     = _xi_0;
    alpha_0  = _alpha_0;
    tau_f    = _tau_f;
    mass     = _mass;
    C    = _C;
    
    step_size = tau_0 / 20;
    steps     = std::ceil((tau_f - tau_0) / step_size);

    SetInitialTemperature();
}

void SimulationParameters::SetInitialTemperature(void)
{
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

        double Ean = std::pow(Lambda_0, 4.0) / (4.0 * PI * PI * alpha_0) * H2Tilde(1.0 / std::sqrt(1.0 + xi_0), mass / Lambda_0);

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
    vec X { alpha_0, Lambda_0, xi_0 };
    pt0 = IntegralJ(2, 0, 1, 0, mass, X) / alpha_0;
    pl0 = IntegralJ(2, 2, 0, 0, mass, X) / alpha_0;
}

double DoubleFactorial(double x)
{
    if (x <= 1.0) return 1.0;
    double y = x * DoubleFactorial(x - 2.0);
    return y;
}

double SimulationParameters::IntegralJ(int n, int r, int q, int s, double mass, vec& X)
{
    double Lambda = X(1);
    double xi = X(2);
    double alpha_L = 1.0 / std::sqrt(1.0 + xi);
    double alpha_T = 1.0;
    double m_bar = mass / Lambda;
    double norm = std::pow(alpha_T, 2* q + 2) * std::pow(alpha_L, r + 1) * std::pow(Lambda, n + s + 2) / (4.0 * PI * PI * DoubleFactorial(2.0 * q));
    
    // PhysRevC.97.054912: Eq. (A8)
    auto Rnrq = [=](double p_bar)
    {
        double w = std::sqrt(alpha_L * alpha_L + std::pow(m_bar / p_bar, 2.0));
        double w3 = w * w * w;
        double z = (alpha_T * alpha_T - alpha_L * alpha_L) / (w * w);
        double z2 {z * z}, z3 {z2 * z}, z4 {z3 * z}, z5 {z4 * z};
        double t = 0;
        if (z == 0) t = 0;
        else if (z < 0) t = std::atanh(std::sqrt(-z)) / std::sqrt(-z);
        else t = std::atan(std::sqrt(z)) / std::sqrt(z);

        if (std::fabs(z) < 0.1)
        {
            if (n == 2 && r == 0 && q == 0)
            {
                return 2.0 * w * (1.0 + z / 3.0 - z2 / 15.0 + z3 / 35.0  - z4 / 63.0 + z5 / 99.0);
            }
            else if (n == 2 && r == 0 && q == 1)
            {
                return 4.0 / w * (1.0 / 3.0 - z / 15.0 + z2 / 35.0 - z3 / 63.0  + z4 / 99.0 - z5 / 143.0);      
            }
            else if (n == 2 && r == 2 && q == 0)
            {
                return 2.0 / w * (1.0 / 3.0 - z / 15.0 + z2 / 35.0 - z3 / 63.0  + z4 / 99.0 - z5 / 143.0);
            }
            else if (n == 2 && r == 2 && q == 1)
            {
                return 4.0 / w3 * (1.0 / 15.0 - 2.0 * z / 35.0 + z2 / 21.0 - 4.0 * z3 / 99.0 + 5.0 * z4 / 143.0 - 2.0 * z5 / 65.0);
            }
            else if (n == 2 && r == 4 && q == 0)
            {
                return 2.0 / w3 * (1.0 / 5.0 - 3.0 * z / 35.0 + z2 / 21.0 - z3 / 33.0 + 3.0 * z4 / 143.0  - z5 / 65.0);
            }
            else if (n == 4 && r == 2 && q == 0)
            {
                return 2.0 * w * (1.0 / 3.0 + z / 15.0 - z2 / 105.0 + z3 / 315.0 - z4 / 693.0 + z5 / 1287.0);
            }
            else if (n == 4 && r == 2 && q == 1)
            {
                return 4.0 / w * (1.0 / 15.0 - 2.0 * z / 105.0 + z2 / 105.0 - 4.0 *z3 / 693.0 + 5.0 * z4 / 1287.0 - 2.0 * z5 / 715.0);
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
            else if (n == 4 && r == 2 && q == 0) return (w * (-1.0 + z + (1.0 + z) * (1.0 + z) * t)) / (4.0 * z);                   // I calculated by hand
            else if (n == 4 && r == 2 && q == 1) return (3.0 + z + (1.0 + z) * (z - 3.0) * t) / (4.0 * z * z * w);
            else if (n == 4 && r == 4 && q == 0) return (-(3.0 + 5.0 * z) + 3.0 * (1.0 + z) * (1.0 + z) * t) / (4.0 * z * z * w);   // I calculated by hand
            else assert("Unsupported choice");
        }
    };

    auto integrand = [=](double p_bar)
    {
        return std::pow(p_bar, n + s + 1 ) * std::pow(1.0 + std::pow(m_bar / p_bar, 2.0), (double)s / 2.0) * Rnrq(p_bar) * std::exp(-std::sqrt(p_bar * p_bar + m_bar * m_bar));
    };
    double result = norm * GausQuad(integrand, 0, inf, 1e-8, 4);
    return result;
}
