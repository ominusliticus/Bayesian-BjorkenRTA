#include "../include/Errors.hpp"
#include "../include/Global_Constants.hpp"
#include "../include/ExactSolution.hpp"


#include <fstream>

int main()
{
    double tau_0 = 0.1;
    double Lambda_0 = 1.0/0.197;
    double xi_0 = 0.0;
    double ul = 100.1;
    double ll = 0.1;
    double mass = 0.0;
    double eta_s = 0.2;
    double steps = 10001;
    double step_size = (ul - ll) / (double)(steps - 1);

    auto soln = exact::ExactSolution(tau_0, Lambda_0, xi_0, ul, ll, mass, eta_s, steps, step_size);
    std::fstream fout("etas0.2_new2.dat", std::fstream::out);
    soln.Run(20, fout);
    return 0;
}