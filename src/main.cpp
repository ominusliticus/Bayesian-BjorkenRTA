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
        default:
        {
            Print(std::cout, "Invalid variable SimulationParameter::type. Please fix.");
            exit(-1234);
        }
    }
    return 0;
}