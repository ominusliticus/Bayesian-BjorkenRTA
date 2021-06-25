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
#include <iomanip>

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    // Print(std::cout, params);

    // exact::Run("./output/exact/temperature_evolution.dat", params);
    // exact::OutputMoments("./output/exact/moments_of_distribution.dat", params);

    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(params);
    
    hydro::ViscousHydroEvolution viscous;
    viscous.RunHydroSimulation(params, hydro::theory::CE);
    viscous.RunHydroSimulation(params, hydro::theory::DNMR);
    return 0;
}