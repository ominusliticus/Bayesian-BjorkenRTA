//
// Author: Kevin Ingles


#include "../include/config.hpp"
#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/HydroTheories.hpp"

#include <chrono>


int main()
{
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);
    // Add logic for runing exact solution: should check if the soluiton with a give parameter set has already run
    // SimulationParameters exact_params("utils/exact_params.txt");
    // if (params != exact_params)
    // {
    //     std::fstream fout("utils/exact_params.txt", std::fstream::out);
    //     fout << exact_params;
    //     fout.close();

    //     exact::ExactSolution hydro_exact;
    //     hydro_exact.Run("output/exact/MCMC_calculation_for_exact.dat", params);
    //     hydro_exact.OutputMoments("output/exact/MCMC_calculation_moments.dat", params);
    // }

    switch (params.type)
    {
        case 0: // Chapman-Enskog hydro
        {
            hydro::ViscousHydroEvolution hydro_ce;
            hydro_ce.RunHydroSimulation(params, hydro::theory::CE);
            hydro_ce.~ViscousHydroEvolution();
            break;
        }
        case 1: // DNMR hydro
        {
            hydro::ViscousHydroEvolution hydro_dnmr;
            hydro_dnmr.RunHydroSimulation(params, hydro::theory::DNMR);
            hydro_dnmr.~ViscousHydroEvolution();
            break;
        }
        case 2: // Anisotropic
        {
            hydro::AnisoHydroEvolution hydro_aniso;
            hydro_aniso.RunHydroSimulation(params);
            hydro_aniso.~AnisoHydroEvolution();
            break;
        }
        case 3: // Alternative anisotropic formulation
        {
            hydro::AltAnisoHydroEvolution hydro_altaniso;
            hydro_altaniso.RunHydroSimulation(params);
            hydro_altaniso.~AltAnisoHydroEvolution();
            break;
        }
        case 4:
        {
            exact::ExactSolution hydro_exact;
            hydro_exact.Run("output/exact/MCMC_calculation_for_exact.dat", params);
            hydro_exact.OutputMoments("output/exact/MCMC_calculation_moments.dat", params);
            break;
        }
        default:
        {
            Print(std::cout, "Invalid variable SimulationParameter::type. Please fix.");
            exit(-1234);
        }
    }
    return 0;
}