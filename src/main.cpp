//
// Author: Kevin Ingles

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/HydroTheories.hpp"

#include <fstream>
#include <fmt/format.h>

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    // Print(std::cout, params);

    // std::fstream fout("./output/etas0.2_new_nonconformal.dat", std::fstream::out);
    // exact::Run(fout, params);
    // fout.close();

    // Print(std::cout, "Calculating moments of distribution function.");
    // fout = std::fstream("./output/moments_of_distribution.dat", std::fstream::out);
    // for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    // {
    //     // Print(std::cout, fmt::format("Evaluating for time {}", tau));
    //     double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
    //     double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
    //     double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
    //     double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
    //     Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    // }
    // fout.close();

    Print(std::cout, "Calculating anistropic hydrodynamic evolution");
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(params);
    return 0;
}