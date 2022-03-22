//
// Author: Kevin Ingles

#include "../include/config.hpp"
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

void generate_data(void);

int main()
{
    // test_blue();
    // test_green();
    // test_magenta();
    // test_maroon();
    // test_orange();
    // test_black();
    // test_cyan();
    generate_data();
    return 0;
}


void test_blue(void)
{
    Print(std::cout, "Starting test case: blue.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        1.64720404472724,
        -0.832036509976845,
        0.654868759801639,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );

    const char* file_name = "./output/temperature_evolution_blue.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_blue.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: blue.\n");
}


void test_green(void)
{
    Print(std::cout, "Starting test case: green.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        0.2111659749634709,
        -0.908162823901714,
        0.000040999483353781,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_green.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_green.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: green.\n");
}


void test_magenta(void)
{
    Print(std::cout, "Starting test case: magenta.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        0.0932423774137986,
        -0.948832296401184,
        2.530008982501650  *pow(10, -8),
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_magenta.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_magenta.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: magenta.\n");
}


void test_maroon(void)
{
    Print(std::cout, "Starting test case: maroon.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        3.44957657894908,
        1208.051523607203,
        0.0780484106646022,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_maroon.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_maroon.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: maroon.\n");
}


void test_orange(void)
{
    Print(std::cout, "Starting test case: orange.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        0.558337869886865,
        -0.987449043604828,
        0.0631735455385675,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_orange.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_orange.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: orange.\n");
}


void test_black(void)
{
    Print(std::cout, "Starting test case: black.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        0.501641547725032,
        0.0,
        0.001064845141251691,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_black.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_black.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: black.\n");
}


void test_cyan(void)
{
    Print(std::cout, "Starting test case: cyan.");
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        0.01 / 0.197,
        0.0,
        2.91164 * pow(10, -14) / 0.197,
        100.1,
        0.2 / 0.197,
        10.0 / (4.0 * PI)
    );
    
    const char* file_name = "./output/temperature_evolution_cyan.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/moments_of_distribution_cyan.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact_soln.TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished test case: cyan.\n");
}

void generate_data(void)
{
    Print(std::cout, "Generating psuedodata."); 
    SimulationParameters params;
    params.SetParameters(
        0.1, 
        1.64720404472724,
        -0.832036509976845,
        0.654868759801639,
        20.1,
        0.2 / 0.197,
        3.0 / (4.0 * PI)
    );

    const char* file_name = "./output/temperature_evolution_blue.dat";
    exact::ExactSolution exact_soln;
    exact_soln.Run(file_name, params);

    Print(std::cout, "Calculating moments of distribution function.");
    std::fstream fout = std::fstream("./output/psuedodata-3_4pi.dat", std::fstream::out);
    for (double tau = params.tau_0; tau <= params.tau_f; tau += 10 * params.step_size)
    {
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact_soln.GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact_soln.GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact_soln.GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact_soln.GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau /*/ exact_soln.TauRelaxation(tau, params) */, new_e_density, new_pL, new_pT, new_peq);
    }
    fout.close();
    Print(std::cout, "Finished generating psuedo_data.\n");
}
