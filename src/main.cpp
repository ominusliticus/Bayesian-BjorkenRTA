//
// Author: Kevin Ingles


#include "../include/config.hpp"
#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/HydroTheories.hpp"
#include "../include/BayesianParameterEstimation.hpp"

#include <chrono>

int main()
{
    auto global_start = std::chrono::steady_clock::now();
    SimulationParameters params("utils/params.txt");

    Print(std::cout, "Anisotropic hydro inference");
    auto aniso_start = std::chrono::steady_clock::now();
    std::string file = "output/aniso_hydro/e2_m=0.200GeV.dat";
    hydro::AnisoHydroEvolution aniso;
    bayes::TestBayesianParameterEstimation<hydro::AnisoHydroEvolution> test_aniso(params, file, aniso);
    test_aniso.RunMCMC(0, 1001, "output/bayes/single_parameter_aniso.dat");
    auto aniso_end = std::chrono::steady_clock::now();

    Print(std::cout, "Viscous CE hydro inference");
    auto ce_start = std::chrono::steady_clock::now();
    file = "output/CE_hydro/e_m=0.200GeV.dat";
    hydro::ViscousHydroEvolution viscous;
    bayes::TestBayesianParameterEstimation<hydro::ViscousHydroEvolution, hydro::theory> test_ce(params, file, viscous);
    test_ce.RunMCMC(0, 1001, "output/bayes/single_parameter_ce.dat", hydro::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    Print(std::cout, "Viscous DNMR hydro inference");
    auto dnmr_start = std::chrono::steady_clock::now();
    file = "output/DNMR_hydro/e_m=0.200GeV.dat";
    bayes::TestBayesianParameterEstimation<hydro::ViscousHydroEvolution, hydro::theory> test_dnmr(params, file, viscous);
    test_dnmr.RunMCMC(0, 1001, "output/bayes/single_parameter_dnmr.dat", hydro::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto global_end = std::chrono::steady_clock::now();

    Print(std::cout, fmt::format("Time for aniso run: {} sec", std::chrono::duration_cast<std::chrono::seconds>(aniso_end - aniso_start).count()));
    Print(std::cout, fmt::format("Time for ce run:    {} sec", std::chrono::duration_cast<std::chrono::seconds>(ce_end - ce_start).count()));
    Print(std::cout, fmt::format("Time for dnmr run:  {} sec", std::chrono::duration_cast<std::chrono::seconds>(dnmr_end - dnmr_start).count()));
    Print(std::cout, fmt::format("Time for total run: {} sec", std::chrono::duration_cast<std::chrono::seconds>(global_end - global_start).count()));
    return 0;
}