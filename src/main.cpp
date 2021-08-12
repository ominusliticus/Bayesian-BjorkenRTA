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
    auto global_start = std::chrono::steady_clock::now();
    SimulationParameters params("utils/params.txt");
    return 0;
}