//
// Author: Kevin Ingles

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"

#include <fstream>
#include <fmt/format.h>


/*
    This test outputs files by the name:
        moments_of_distribution_{color}.dat
    and is meant to be compared to the plots in Figs. (4) - (7)
    of Chattopadhyay et al. (in progress)
*/

void test_blue   (void);
void test_green  (void);
void test_magenta(void);
void test_maroon (void);
void test_orange (void);
void test_black  (void);
void test_cyan   (void);

int main()
{
    test_blue();
    test_green();
    test_magenta();
    test_maroon();
    test_orange();
    test_black();
    test_cyan();
    return 0;
}


void test_blue(void)
{
    Print(std::cout, "Starting test case: blue.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 1.64720404472724;
    params.alpha_0   = 0.654868759801639;
    params.xi_0      = -0.832036509976845;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);

    std::fstream fout("./output/temperature_evolution_blue.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_blue.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: blue.\n");
}


void test_green(void)
{
    Print(std::cout, "Starting test case: green.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 0.2111659749634709;
    params.alpha_0   = 0.000040999483353781;
    params.xi_0      = -0.908162823901714;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_green.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_green.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: green.\n");
}


void test_magenta(void)
{
    Print(std::cout, "Starting test case: magenta.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 0.0932423774137986;
    params.alpha_0   = 2.530008982501650  *pow(10, -8);
    params.xi_0      = -0.948832296401184;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_magenta.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_magenta.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: magenta.\n");
}


void test_maroon(void)
{
    Print(std::cout, "Starting test case: maroon.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 3.44957657894908;
    params.alpha_0   = 0.0780484106646022;
    params.xi_0      = 1208.051523607203;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_maroon.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_maroon.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: maroon.\n");
}


void test_orange(void)
{
    Print(std::cout, "Starting test case: orange.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 0.558337869886865;
    params.alpha_0   = 0.0631735455385675;
    params.xi_0      = -0.987449043604828;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_orange.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_orange.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: orange.\n");
}


void test_black(void)
{
    Print(std::cout, "Starting test case: black.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 0.501641547725032;
    params.alpha_0   = 0.001064845141251691;
    params.xi_0      = 0.0;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_black.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_black.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: black.\n");
}


void test_cyan(void)
{
    Print(std::cout, "Starting test case: cyan.");
    SimulationParameters params;
    params.tau_0     = 0.1;
    params.Lambda_0  = 0.01 / 0.197;
    params.alpha_0   = 2.91164 * pow(10, -14) / 0.197;
    params.xi_0      = 0.0;
    params.ul        = 100.1;
    params.ll        = 0.1;
    params.mass      = 0.2 / 0.197;
    params.eta_s     = 10 / (4.0 * PI);
    params.steps     = 10001;
    params.step_size = (100.1 - 0.1) / 10001;
    params.D.resize(10001);
    
    std::fstream fout("./output/temperature_evolution_cyan.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution_cyan.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: cyan.\n");
}