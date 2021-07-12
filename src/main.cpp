//
// Author: Kevin Ingles

// Uncomment to enable parallel evaluations in ExactSolution.cpp
// Test show that it is not faster though.
#define USE_PARALLEL

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/HydroTheories.hpp"

#include <chrono>

int main()
{
    // TO DO: clean up benchmarking
    auto start = std::chrono::steady_clock::now();
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);

    auto eaxct_start = std::chrono::steady_clock::now();
    exact::ExactSolution exact_soln;
    exact_soln.Run("./output/exact/temperature_evolution.dat", params);
    auto exact_end = std::chrono::steady_clock::now();
    
    exact_soln.OutputMoments(fmt::format("./output/exact/moments_of_distribution_m={:.3f}GeV.dat", 0.197 * params.mass).data(), params);

    double t0  = params.tau_0;
    double pt0 = exact_soln.GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact_soln.GetMoments(t0, params, exact::Moment::PL);
    params.SetParameter("pt0",pt0);
    params.SetParameter("pl0", pl0);
    Print(std::cout, params);

    auto aniso_start = std::chrono::steady_clock::now();
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(params);
    auto aniso_end = std::chrono::steady_clock::now();

    auto altaniso_start = std::chrono::steady_clock::now();
    hydro::AltAnisoHydroEvolution altaniso;
    altaniso.RunHydroSimulation(params);
    auto altaniso_end = std::chrono::steady_clock::now();
    
    hydro::ViscousHydroEvolution viscous;
    auto ce_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(params, hydro::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    auto dnmr_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(params, hydro::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();
    Print(std::cout, fmt::format("Time for exact solution: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(exact_end - eaxct_start).count()));
    Print(std::cout, fmt::format("Time for ansio solution: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(aniso_end - aniso_start).count()));
    Print(std::cout, fmt::format("Time for altaniso solution: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(altaniso_end - altaniso_start).count()));
    Print(std::cout, fmt::format("Time for ce solution: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(ce_end - ce_start).count()));
    Print(std::cout, fmt::format("Time for dnmr solution: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(dnmr_end - dnmr_start).count()));
    Print(std::cout, fmt::format("Total simulation duraion: {} sec", (long long int)std::chrono::duration_cast<std::chrono::seconds>(end - start).count()));
    return 0;
}