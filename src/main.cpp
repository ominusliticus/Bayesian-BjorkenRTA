//
// Author: Kevin Ingles

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"
#include "../include/HydroTheories.hpp"

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);

    exact::Run("./output/exact/temperature_evolution.dat", params);
    exact::OutputMoments("./output/exact/moments_of_distribution.dat", params);

    double t0  = params.tau_0;
    double pt0 = exact::GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact::GetMoments(t0, params, exact::Moment::PL);
    params.pt0 = pt0;
    params.pl0 = pl0;


    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(params);
    
    hydro::ViscousHydroEvolution viscous;
    viscous.RunHydroSimulation(params, hydro::theory::CE);
    viscous.RunHydroSimulation(params, hydro::theory::DNMR);
    return 0;
}