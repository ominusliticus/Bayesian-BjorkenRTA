//
// Author: Kevin Ingles

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"

#include <fstream>
#include <fmt/format.h>

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);

    std::fstream fout("./output/etas0.2_new_nonconformal.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    fout = std::fstream("./output/e_density_comparison.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= 50.1; tau += 1.0)
    {
        std::fstream fwrite(fmt::format("./output/extact_solution_{:.{}f}.dat", tau, 1), std::fstream::out);
        Print(std::cout, fmt::format("Evaluating for time {}", tau));
        for (double p = 0.0; p < 3.0; p += 0.01)
        {
            double fexact = GausQuad([](double theta, double p, double tau, SP& params){
                return exact::EaxctDisbtribution(theta, p, tau, params);
            }, 0, PI / 2, eps, 2, p, tau, params);
            Print(fwrite, tau, p, fexact);
        }
        fwrite.close();

        double old_e_density = exact::GetMoments(tau, params);
        double new_e_density = exact::GetMoments2(tau, params);
        Print(fout,tau, old_e_density, new_e_density);
    }
    fout.close();
    return 0;
}