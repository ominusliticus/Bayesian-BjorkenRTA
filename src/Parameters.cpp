//
// Author: Kevin Ingles

#include "../include/Parameters.hpp"

#include <fstream>
#include <sstream>

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
            if (var_name.compare("tau_0") == 0)          buffer >> tau_0;
            else if (var_name.compare("Lambda_0") == 0)  buffer >> Lambda_0;
            else if (var_name.compare("xi_0") == 0)      buffer >> xi_0;
            else if (var_name.compare("alpha_0") == 0)   buffer >> alpha_0;
            else if (var_name.compare("ul") == 0)        buffer >> ul;
            else if (var_name.compare("ll") == 0)        buffer >> ll;
            else if (var_name.compare("mass") == 0)      buffer >> mass;
            else if (var_name.compare("eta_s") == 0)     buffer >> eta_s;
            else if (var_name.compare("steps") == 0)     buffer >> steps;
        } // end else
    } // end while(!fin.eof())
    step_size = (ul - ll) / (double) (steps - 1);
    D.resize(steps);
} // end SimulationParameters::SimulationParameters(...)
// -----------------------------------------

SimulationParameters::~SimulationParameters()
{
    D.clear();
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
    Print(out, "D.size() ", params.D.size());
    return out;
}