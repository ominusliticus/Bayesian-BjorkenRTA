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

long int integration_timer;
long int matrix_op_timer;
long int loop_timer; 

int main()
{
    int steps = 100;
    auto global_start = std::chrono::steady_clock::now();
    SimulationParameters params("utils/params.txt");

    std::vector<double> tau_f = {
        10.0,
        11.0,
        12.0,
        13.0, 
        14.0,
        15.0,
    };

    std::vector<double> e_data = {
        0.0540145897416580,
        0.0484834026497000,
        0.0439132833714083,
        0.0400780128208830,
        0.0368168554634228,
        0.0340121294241507,
    };

    std::vector<double> pi_data = {
        (2.0 / 3.0) * (0.0070858408559075 - 0.0188765110236328),
        (2.0 / 3.0) * (0.0065824225885789 - 0.0166451988062845),
        (2.0 / 3.0) * (0.0061427019034975 - 0.0148268586016080),
        (2.0 / 3.0) * (0.0057549973900186 - 0.0133214024234779),
        (2.0 / 3.0) * (0.0054104615083681 - 0.0120580918132368),
        (2.0 / 3.0) * (0.0051021367829928 - 0.0109854896009394)
    };

    std::vector<double> Pi_data = {
        (0.0070858408559075 + 2.0 * 0.0188765110236328) / 3.0 - 0.0145882073981164,
        (0.0065824225885789 + 2.0 * 0.0166451988062845) / 3.0 - 0.0129986574326894,
        (0.0061427019034975 + 2.0 * 0.0148268586016080) / 3.0 - 0.0116930920287641,
        (0.0057549973900186 + 2.0 * 0.0133214024234779) / 3.0 - 0.0106031985535975,
        (0.0054104615083681 + 2.0 * 0.0120580918132368) / 3.0 - 0.0096814221548396,
        (0.0051021367829928 + 2.0 * 0.0109854896009394) / 3.0 - 0.0088921407933220
    };

    // Print(std::cout, "Anisotropic hydro inference");
    // auto aniso_start = std::chrono::steady_clock::now();
    // hydro::AnisoHydroEvolution aniso;
    // bayes::TestBayesianParameterEstimation<hydro::AnisoHydroEvolution> test_aniso(params, file, aniso);
    // test_aniso.RunMCMC(0, steps, "output/bayes/single_parameter_aniso.dat");
    // auto aniso_end = std::chrono::steady_clock::now();

    Print(std::cout, "Viscous CE hydro inference");
    auto ce_start = std::chrono::steady_clock::now();
    hydro::ViscousHydroEvolution viscous;
    bayes::TestBayesianParameterEstimation<hydro::ViscousHydroEvolution, hydro::theory> test_ce(params, tau_f, e_data, pi_data, Pi_data, e_data, pi_data, Pi_data, viscous);
    test_ce.RunMCMC(0, steps, "output/bayes/single_parameter_ce.dat", hydro::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    // Print(std::cout, "Viscous DNMR hydro inference");
    // auto dnmr_start = std::chrono::steady_clock::now();
    // file = "output/DNMR_hydro/e_m=0.200GeV.dat";
    // bayes::TestBayesianParameterEstimation<hydro::ViscousHydroEvolution, hydro::theory> test_dnmr(params, file, viscous);
    // test_dnmr.RunMCMC(0, steps, "output/bayes/single_parameter_dnmr.dat", hydro::theory::DNMR);
    // auto dnmr_end = std::chrono::steady_clock::now();

    // Print(std::cout, "Alternative anisotropic inference");
    // auto altaniso_start = std::chrono::steady_clock::now();
    // file = "output/aniso_hydro/e_m=0.200GeV.dat";
    // hydro::AltAnisoHydroEvolution altaniso;
    // bayes::TestBayesianParameterEstimation<hydro::AltAnisoHydroEvolution> test_altaniso(params, file, altaniso);
    // test_altaniso.RunMCMC(0, steps, "output/bayes/singles_parameter_altaniso.dat");
    // auto altaniso_end = std::chrono::steady_clock::now();

    auto global_end = std::chrono::steady_clock::now();

    // Print(std::cout, fmt::format("Time for aniso run:     {} sec", std::chrono::duration_cast<std::chrono::seconds>(aniso_end - aniso_start).count()));
    Print(std::cout, fmt::format("Time for ce run:        {} sec", std::chrono::duration_cast<std::chrono::seconds>(ce_end - ce_start).count()));
    // Print(std::cout, fmt::format("Time for dnmr run:      {} sec", std::chrono::duration_cast<std::chrono::seconds>(dnmr_end - dnmr_start).count()));
    // Print(std::cout, fmt::format("Time for alt aniso run: {} sec", std::chrono::duration_cast<std::chrono::seconds>(altaniso_end - altaniso_start).count()));
    Print(std::cout, fmt::format("Time for total run:     {} sec", std::chrono::duration_cast<std::chrono::seconds>(global_end - global_start).count()));
    return 0;
}