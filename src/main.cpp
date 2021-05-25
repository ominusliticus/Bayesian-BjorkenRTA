#include "../include/Errors.hpp"
#include "../include/GlobalConstants.hpp"
#include "../include/ExactSolution.hpp"
#include "../include/Parameters.hpp"


#include <fstream>

int main()
{
    Print(std::cout, "Calculating exact solution:\n");
    SimulationParameters params("utils/params.txt");
    Print(std::cout, params);

    // ExactSolution has a global static vector name D which needs to be 
    // declared static to avoid multiple copies, but also needs to be 
    // resized in the int main() function
    std::fstream fout("etas0.2_new_nonconformal.dat", std::fstream::out);
    exact::Run(fout, params);
    fout.close();
    return 0;
}