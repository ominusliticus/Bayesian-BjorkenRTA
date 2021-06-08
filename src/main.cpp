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
    for (double tau = params.ll; tau <= params.ul; tau += 100 * params.step_size)
    {
        Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments2(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments2(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments2(tau, params, exact::Moment::PT);
        Print(fout, tau, new_e_density, new_pL, new_pT);
    }
    fout.close();
    return 0;
}