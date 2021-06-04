//
// Author: Kevin Ingles

#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"
#include "../include/Integration.hpp"

#include <fstream>
#include <fmt/format.h>

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);

    std::fstream fout("./output/etas0.2_new_nonconformal.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();

    Print(std::cout, "Calculating moments of distribution function.");
    fout = std::fstream("./output/moments_of_distribution.dat", std::fstream::out);
    for (double tau = params.ll; tau <= params.ul; tau += 10 * params.step_size)
    {
<<<<<<< HEAD
        // Print(std::cout, fmt::format("Evaluating for time {}", tau));
        double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
        double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
        double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
        double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
        Print(fout, tau / exact::TauRelaxation(tau, params), new_e_density, new_pL, new_pT, new_peq);
=======
        std::fstream fwrite1(fmt::format("./output/extact_solution_tuple_{:.{}f}.dat", tau, 1), std::fstream::out);
        std::fstream fwrite2(fmt::format("./output/exact_solution_tuple_theta_intergrate_{:.{}f}.dat", tau, 1), std::fstream::out);
        Print(std::cout, fmt::format("Evaluating for time {}", tau));
        // for (double w = -3.0; w <= 3.0; w += 0.02)
        //     for (double pT = -3.0; pT <= 3.0; pT += 0.02)
        //         Print(fwrite, w, pT, exact::EaxctDistribution(w, pT, tau, params));
        // fwrite.close();

        for (double p = 0.0; p <= 3.0; p += 0.01)
        {
            auto [first1, second1] = exact::EaxctDistributionTuple(0, p, tau, params);
            Print(fwrite1, p, first1, second1);
            
            auto [first2, second2] = exact::ThetaIntegratedExactDistributionTuple(p, tau, params);
            Print(fwrite2, p, first2, second2);
        }
        fwrite1.close();
        fwrite2.close();

        double old_e_density = exact::GetMoments(tau, params);
        double new_e_density = exact::GetMoments2(tau, params);
        Print(fout, tau, old_e_density, new_e_density);
>>>>>>> Initial commit to add_sampler branch
    }
    fout.close();
    return 0;
}