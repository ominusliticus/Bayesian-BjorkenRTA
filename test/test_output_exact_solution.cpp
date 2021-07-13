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
    exact::ExactSolution exact_soln;
    exact_soln.Run(fout, params);
    fout.close();

    for (double tau = params.ll; tau <= params.ul; tau += 100 * params.step_size)
    {
        Print(std::cout, fmt::format("Evaluating for time {}", tau)); 
        std::fstream fwrite(fmt::format("./output/extact_solution_{:.{}f}.dat", tau, 1), std::fstream::out);        
        for (double w = -3.0; w <= 3.0; w += 0.02)
            for (double pT = -3.0; pT <= 3.0; pT += 0.02)
                Print(fwrite, w, pT, exact_soln.EaxctDistribution(w, pT, tau, params));
        fwrite.close();
    }
    return 0;
}