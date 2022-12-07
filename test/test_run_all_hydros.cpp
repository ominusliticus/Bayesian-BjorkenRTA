//
// Author: Kevin Ingles

/* This file runs the various hydro implementations as well as the RTA solution
    which is needed to calculate the initial conditions of the pressures (for
   now). The output files will correspond to plots I will include in my paper
*/

// Uncomment to enable parallel evaluations in ExactSolution.cpp
#include "../include/config.hpp"
#include "../include/Errors.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/HydroTheories.hpp"
#include "../include/Integration.hpp"
#include "../include/Parameters.hpp"

#include <chrono>
#include <filesystem>
#include <string>

void Run1MeV(void);
void Run50MeV(void);
void Run200MeV(void);
void Run1GeV(void);

bool compare_floats_same(double, double);
bool compare_output_and_expected(std::string);

static double                tolerance         = 1e-8;
static std::filesystem::path cwd               = std::filesystem::current_path() / "output/test";
static std::string           output_dir_string = cwd.string();
static const char*           output_dir        = cwd.c_str();
static const char*           file_suffix       = ".dat";

// Hydro strings
[[maybe_unused]] static std::string ce_string   = "ce_";
[[maybe_unused]] static std::string dnmr_string = "dnmr_";
[[maybe_unused]] static std::string mis_string  = "mis_";
[[maybe_unused]] static std::string vah_string  = "vah_";
[[maybe_unused]] static std::string mvah_string = "mvah_";

// Observables strings
[[maybe_unused]] static std::string bulk_string   = "bulk_";
[[maybe_unused]] static std::string energy_string = "e_";
[[maybe_unused]] static std::string shear_string  = "shear_";

int main()
{
    Run1MeV();
    Run50MeV();
    Run200MeV();
    Run1GeV();
    return 0;
}

void Run1MeV(void)
{
    auto start = std::chrono::steady_clock::now();
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params;
    params.SetParameters(0.1, 12.4991, 6.0677, 0.0090, 12.1, 0.001 / 0.197, 3.0 / (4.0 * PI));
    Print(std::cout, params);

    auto                 eaxct_start = std::chrono::steady_clock::now();
    exact::ExactSolution exact_soln(params);
    exact_soln.Run(params);
    auto exact_end = std::chrono::steady_clock::now();

    exact_soln.OutputMoments(output_dir, params);

    Print(std::cout, "Setting iniital conditions for pl and pt.");
    double t0  = params.tau_0;
    double pt0 = exact_soln.GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact_soln.GetMoments(t0, params, exact::Moment::PL);
    params.SetParameter("pt0", pt0);
    params.SetParameter("pl0", pl0);
    Print(std::cout, params);

    auto                       aniso_start = std::chrono::steady_clock::now();
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(output_dir, params);
    auto aniso_end = std::chrono::steady_clock::now();

    auto                          altaniso_start = std::chrono::steady_clock::now();
    hydro::AltAnisoHydroEvolution altaniso;
    altaniso.RunHydroSimulation(output_dir, params);
    auto altaniso_end = std::chrono::steady_clock::now();

    hydro::ViscousHydroEvolution viscous;
    auto                         ce_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    auto dnmr_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto mis_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::MIS);
    auto mis_end = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();

    Print(std::cout,
          fmt::format(
              "Time for exact solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(exact_end - eaxct_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for ansio solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(aniso_end - aniso_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for altaniso solution: {} sec",
              static_cast<long long int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             altaniso_end - altaniso_start)
                                             .count())));
    Print(
        std::cout,
        fmt::format(
            "Time for ce solution: {} sec",
            static_cast<long long int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(ce_end - ce_start).count())));
    Print(
        std::cout,
        fmt::format("Time for dnmr solution: {} sec",
                    static_cast<long long int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(dnmr_end - dnmr_start)
                            .count())));
    Print(std::cout,
          fmt::format("Time for mis solution: {} sec",
                      static_cast<long long int>(
                          std::chrono::duration_cast<std::chrono::milliseconds>(mis_end - mis_start)
                              .count())));
    Print(std::cout,
          fmt::format(
              "Total simulation duraion: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())));

    std::string mass_string = "m=0.001GeV";
    if (compare_output_and_expected(mass_string))
        Print(std::cout, "Test hdyros: Mass = 1 MeV: \033[1;32mPASSES!\033[0m");
    else Print(std::cout, "Test hydros: Mass = 1 MeV: \033[1;31mFAILS!\033[0m");
}

void Run50MeV(void)
{
    auto start = std::chrono::steady_clock::now();
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params;
    params.SetParameters(0.1, 12.4991, 6.0677, 0.0090, 12.1, 0.050 / 0.197, 3.0 / (4.0 * PI));
    Print(std::cout, params);

    auto                 eaxct_start = std::chrono::steady_clock::now();
    exact::ExactSolution exact_soln(params);
    exact_soln.Run(params);
    auto exact_end = std::chrono::steady_clock::now();

    exact_soln.OutputMoments(output_dir, params);

    Print(std::cout, "Setting iniital conditions for pl and pt.");
    double t0  = params.tau_0;
    double pt0 = exact_soln.GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact_soln.GetMoments(t0, params, exact::Moment::PL);
    params.SetParameter("pt0", pt0);
    params.SetParameter("pl0", pl0);
    Print(std::cout, params);

    auto                       aniso_start = std::chrono::steady_clock::now();
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(output_dir, params);
    auto aniso_end = std::chrono::steady_clock::now();

    auto                          altaniso_start = std::chrono::steady_clock::now();
    hydro::AltAnisoHydroEvolution altaniso;
    altaniso.RunHydroSimulation(output_dir, params);
    auto altaniso_end = std::chrono::steady_clock::now();

    hydro::ViscousHydroEvolution viscous;
    auto                         ce_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    auto dnmr_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto mis_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::MIS);
    auto mis_end = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();

    Print(std::cout,
          fmt::format(
              "Time for exact solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(exact_end - eaxct_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for ansio solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(aniso_end - aniso_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for altaniso solution: {} sec",
              static_cast<long long int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             altaniso_end - altaniso_start)
                                             .count())));
    Print(
        std::cout,
        fmt::format(
            "Time for ce solution: {} sec",
            static_cast<long long int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(ce_end - ce_start).count())));
    Print(
        std::cout,
        fmt::format("Time for dnmr solution: {} sec",
                    static_cast<long long int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(dnmr_end - dnmr_start)
                            .count())));
    Print(std::cout,
          fmt::format("Time for mis solution: {} sec",
                      static_cast<long long int>(
                          std::chrono::duration_cast<std::chrono::milliseconds>(mis_end - mis_start)
                              .count())));
    Print(std::cout,
          fmt::format(
              "Total simulation duraion: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())));

    std::string mass_string = "m=0.050GeV";
    if (compare_output_and_expected(mass_string))
        Print(std::cout, "Test hdyros: Mass = 1 MeV: \033[1;32mPASSES!\033[0m");
    else Print(std::cout, "Test hydros: Mass = 1 MeV: \033[1;31mFAILS!\033[0m");
}

void Run200MeV(void)
{
    auto start = std::chrono::steady_clock::now();
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params;
    params.SetParameters(0.1, 12.4991, 6.0677, 0.0090, 12.1, 0.2 / 0.197, 3.0 / (4.0 * PI));
    Print(std::cout, params);

    auto                 eaxct_start = std::chrono::steady_clock::now();
    exact::ExactSolution exact_soln(params);
    exact_soln.Run(params);
    auto exact_end = std::chrono::steady_clock::now();

    exact_soln.OutputMoments(output_dir, params);

    Print(std::cout, "Setting iniital conditions for pl and pt.");
    double t0  = params.tau_0;
    double pt0 = exact_soln.GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact_soln.GetMoments(t0, params, exact::Moment::PL);
    params.SetParameter("pt0", pt0);
    params.SetParameter("pl0", pl0);
    Print(std::cout, params);

    auto                       aniso_start = std::chrono::steady_clock::now();
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(output_dir, params);
    auto aniso_end = std::chrono::steady_clock::now();

    auto                          altaniso_start = std::chrono::steady_clock::now();
    hydro::AltAnisoHydroEvolution altaniso;
    altaniso.RunHydroSimulation(output_dir, params);
    auto altaniso_end = std::chrono::steady_clock::now();

    hydro::ViscousHydroEvolution viscous;
    auto                         ce_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    auto dnmr_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto mis_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::MIS);
    auto mis_end = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();

    Print(std::cout,
          fmt::format(
              "Time for exact solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(exact_end - eaxct_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for ansio solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(aniso_end - aniso_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for altaniso solution: {} sec",
              static_cast<long long int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             altaniso_end - altaniso_start)
                                             .count())));
    Print(
        std::cout,
        fmt::format(
            "Time for ce solution: {} sec",
            static_cast<long long int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(ce_end - ce_start).count())));
    Print(
        std::cout,
        fmt::format("Time for dnmr solution: {} sec",
                    static_cast<long long int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(dnmr_end - dnmr_start)
                            .count())));
    Print(std::cout,
          fmt::format("Time for mis solution: {} sec",
                      static_cast<long long int>(
                          std::chrono::duration_cast<std::chrono::milliseconds>(mis_end - mis_start)
                              .count())));
    Print(std::cout,
          fmt::format(
              "Total simulation duraion: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())));

    std::string mass_string = "m=0.200GeV";
    if (compare_output_and_expected(mass_string))
        Print(std::cout, "Test hdyros: Mass = 1 MeV: \033[1;32mPASSES!\033[0m");
    else Print(std::cout, "Test hydros: Mass = 1 MeV: \033[1;31mFAILS!\033[0m");
}

void Run1GeV(void)
{
    auto start = std::chrono::steady_clock::now();
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params;
    params.SetParameters(0.1, 12.4991, 6.0677, 0.0090, 12.1, 1.0 / 0.197, 3.0 / (4.0 * PI));
    Print(std::cout, params);

    auto                 eaxct_start = std::chrono::steady_clock::now();
    exact::ExactSolution exact_soln(params);
    exact_soln.Run(params);
    auto exact_end = std::chrono::steady_clock::now();

    exact_soln.OutputMoments(output_dir, params);

    Print(std::cout, "Setting iniital conditions for pl and pt.");
    double t0  = params.tau_0;
    double pt0 = exact_soln.GetMoments(t0, params, exact::Moment::PT);
    double pl0 = exact_soln.GetMoments(t0, params, exact::Moment::PL);
    params.SetParameter("pt0", pt0);
    params.SetParameter("pl0", pl0);
    Print(std::cout, params);

    auto                       aniso_start = std::chrono::steady_clock::now();
    hydro::AnisoHydroEvolution aniso;
    aniso.RunHydroSimulation(output_dir, params);
    auto aniso_end = std::chrono::steady_clock::now();

    auto                          altaniso_start = std::chrono::steady_clock::now();
    hydro::AltAnisoHydroEvolution altaniso;
    altaniso.RunHydroSimulation(output_dir, params);
    auto altaniso_end = std::chrono::steady_clock::now();

    hydro::ViscousHydroEvolution viscous;
    auto                         ce_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::CE);
    auto ce_end = std::chrono::steady_clock::now();

    auto dnmr_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::DNMR);
    auto dnmr_end = std::chrono::steady_clock::now();

    auto mis_start = std::chrono::steady_clock::now();
    viscous.RunHydroSimulation(output_dir, params, hydro::ViscousHydroEvolution::theory::MIS);
    auto mis_end = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();

    Print(std::cout,
          fmt::format(
              "Time for exact solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(exact_end - eaxct_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for ansio solution: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(aniso_end - aniso_start)
                      .count())));
    Print(std::cout,
          fmt::format(
              "Time for altaniso solution: {} sec",
              static_cast<long long int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             altaniso_end - altaniso_start)
                                             .count())));
    Print(
        std::cout,
        fmt::format(
            "Time for ce solution: {} sec",
            static_cast<long long int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(ce_end - ce_start).count())));
    Print(
        std::cout,
        fmt::format("Time for dnmr solution: {} sec",
                    static_cast<long long int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(dnmr_end - dnmr_start)
                            .count())));
    Print(std::cout,
          fmt::format("Time for mis solution: {} sec",
                      static_cast<long long int>(
                          std::chrono::duration_cast<std::chrono::milliseconds>(mis_end - mis_start)
                              .count())));
    Print(std::cout,
          fmt::format(
              "Total simulation duraion: {} sec",
              static_cast<long long int>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())));

    std::string mass_string = "m=1.000GeV";
    if (compare_output_and_expected(mass_string))
        Print(std::cout, "Test hdyros: Mass = 1 MeV: \033[1;32mPASSES!\033[0m");
    else Print(std::cout, "Test hydros: Mass = 1 MeV: \033[1;31mFAILS!\033[0m");
}

bool compare_floats_same(double lhs, double rhs)
{
    auto percent_difference = std::abs(lhs - rhs) / std::abs(rhs);
    return percent_difference < tolerance;
}

bool compare_output_and_expected(std::string mass_string)
{
    std::vector<std::string> output_files = {
        (ce_string + bulk_string + mass_string + file_suffix),
        (ce_string + energy_string + mass_string + file_suffix),
        (ce_string + shear_string + mass_string + file_suffix),
        (dnmr_string + bulk_string + mass_string + file_suffix),
        (dnmr_string + energy_string + mass_string + file_suffix),
        (dnmr_string + shear_string + mass_string + file_suffix),
        (mis_string + bulk_string + mass_string + file_suffix),
        (mis_string + energy_string + mass_string + file_suffix),
        (mis_string + shear_string + mass_string + file_suffix),
        (vah_string + bulk_string + mass_string + file_suffix),
        (vah_string + energy_string + mass_string + file_suffix),
        (vah_string + shear_string + mass_string + file_suffix),
        (mvah_string + bulk_string + mass_string + file_suffix),
        (mvah_string + bulk_string + mass_string + file_suffix),
        (mvah_string + energy_string + mass_string + file_suffix),
    };

    for (const auto& file_string : output_files)
    {
        std::fstream calculation_file(cwd / file_string, std::fstream::in);
        std::fstream comparison_file(cwd / ("comp_" + file_string), std::fstream::in);

        double tau_calc, observable_calc;
        double tau_comp, observable_comp;
        bool   comparison_result = true;
        while ((calculation_file >> tau_calc >> observable_calc)
               && (comparison_file >> tau_comp >> observable_comp))
        {
            comparison_result &= compare_floats_same(observable_comp, observable_calc);
            if (!comparison_result) [[unlikely]]
            {
                Print(std::cout, "Test failed while comparing", file_string);
                Print(std::cout,
                      "To debug, please note that you will want to compare ",
                      "the output of the exact ",
                      "solutions, which is not done in this test case");
                return comparison_result;
            }
        }    // while reading files
    }        // for file list of files
    return true;
}

