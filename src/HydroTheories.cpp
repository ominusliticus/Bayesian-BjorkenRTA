//
// Author: Kevin Ingles

#include "../include/HydroTheories.hpp"

#include <gsl/gsl_linalg.h>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace hydro
{
    const double tol = 1e-15;
    const int max_depth = 1;
    // utility fuction for quick exponentiation
    double pow(double base, double exp)
    {
        if (base < 0 )
            return std::exp(exp * std::log(-base)) * std::cos(exp * PI);
        else 
            return std::exp(exp * std::log(base));;
    }
    // -------------------------------------

    double doubleFactorial(int k)
    {
        if (k <= 1)
            return 1.0;
        double result = (double)k * doubleFactorial(k - 2);
        return result;
    }
    // -------------------------------------

    // Viscous struct implementation


    // Anisotropic struct implementation
    void AnisoHydroEvolution::RunHydroSimulation(SP& params)
    {
        double t0 = params.tau_0;
        double dt = t0 / 20.0; //params.step_size;

        // for setprecision of t output
        int decimal = - (int) std::log10(dt);

        // Opening output files
        std::fstream e_plot("output/aniso_hydro/e.dat", std::ios::out);
        std::fstream pl_plot("output/aniso_hydro/pl.dat", std::ios::out);
        std::fstream pt_plot("output/aniso_hydro/pt.dat", std::ios::out);
        if (!e_plot.is_open() && !pl_plot.is_open() && !pt_plot.is_open())
        {
            Print_Error(std::cerr, "AnisoHydroEvolution::RunHydroSimulation: Failed to open output files.");
            Print_Error(std::cerr, "Pleae make sure the folder ./output/aniso_hydro/ exists.");
            exit(-3333);
        }
        else
        {
            Print(e_plot , std::fixed, std::setprecision(decimal + 2));
            Print(pl_plot, std::fixed, std::setprecision(decimal + 2));
            Print(pt_plot, std::fixed, std::setprecision(decimal + 2));
        }
        
        // Initialize simulation
        T0         = 0.5 / 0.197;       // Note that the temperature is already in units fm^{-1}
        double m   = params.mass;       // Note theat the mass in already in units fm^{-1}
        double z0  = m / T0;
        double e0;                      // Equilibrium energy density
        if (m == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
        else e0 = 3.0 * pow(T0, 4.0) / (PI * PI) * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

        // Note: all dynamic variables are declared as struct memebrs variables
        e1  = e0;
        if (m == 0)
        {
            pt1 = e0 / 4.0;
            pl1 = e0 / 2.0;
        }
        else
        {
            pt1 = params.pt0;
            pl1 = params.pl0;
        }
        
        // Begin simulation 
        TransportCoefficients tc;
        double t;
        for (int n = 0; n < 25; n++)
        {
            t = t0 + n * dt;
            Print(e_plot, t, e1 );
            Print(pl_plot, t, pl1);
            Print(pt_plot, t, pt1);

            // RK4 with updating anisotropic variables
            // Note all dynamic variables are declared as member variables

            Print(std::cout, fmt::format("t = {:.5f}\te = {:.5f}\tpt = {:.5f}\tpl = {:.5f}", t, e1, pt1, pl1));
            
            // First order
            tc = CalculateTransportCoefficients(e1, pt1, pl1, t, params);
            de1  = dt *  dedt(e1, pl1, t);
            dpt1 = dt * dptdt(e1, pt1, pl1, t, tc);
            dpl1 = dt * dpldt(e1, pt1, pl1, t, tc);

            e2  = e1  + de1  / 2.0;
            pt2 = pt1 + dpt1 / 2.0;
            pl2 = pl1 + dpl1 / 2.0;

            // Second order
            tc = CalculateTransportCoefficients(e2, pt2, pl2, t + dt / 2.0, params);
            de2  = dt *  dedt(e2, pl2, t  + dt / 2.0);
            dpt2 = dt * dptdt(e2, pt2, pl2, t + dt / 2.0, tc);
            dpl2 = dt * dpldt(e2, pt2, pl2, t + dt / 2.0, tc);

            e3  = e1  + de2  / 2.0;
            pt3 = pt1 + dpt2 / 2.0;
            pl3 = pl1 + dpl2 / 2.0;

            // Third order
            tc = CalculateTransportCoefficients(e3, pt3, pl3, t + dt / 2.0, params);
            de3  = dt *  dedt(e3, pl3, t + dt / 2.0);
            dpt3 = dt * dptdt(e3, pt3, pl3, t + dt / 2.0, tc);
            dpl3 = dt * dpldt(e3, pt3, pl3, t + dt / 2.0, tc);
            
            e4  = e1  + de3;
            pt4 = pt1 + dpt3;
            pl4 = pl1 + dpl3;

            // Fourth order
            tc = CalculateTransportCoefficients(e4, pt4, pl4, t, params);
            de4  = dt *  dedt(e4, pl4, t + dt);
            dpt4 = dt * dptdt(e4, pt4, pl4, t + dt, tc);
            dpl4 = dt * dpldt(e4, pt4, pl4, t + dt, tc);

            e1  += (de1  + 2.0 * de2  + 2.0 * de3  +  de4) / 6.0;
            pt1 += (dpt1 + 2.0 * dpt2 + 2.0 * dpt3 + dpt4) / 6.0;
            pl1 += (dpl1 + 2.0 * dpl2 + 2.0 * dpl3 + dpl4) / 6.0;
            Print(std::cout);

        } // End simulation loop
    }
    // -------------------------------------


    double AnisoHydroEvolution::EquilibriumEnergyDensity(double temp, SP& params)
    {
        double z = params.mass / temp;
        if (z == 0) return 3.0 * pow(temp, 4.0) / (PI * PI);
        else return 3.0 * pow(temp, 4.0) / (PI * PI) * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
    }
    // -------------------------------------



    double AnisoHydroEvolution::InvertEnergyDensity(double e, SP& params)
    {
        double x1, x2, mid;
        double T_min = .001/.197;
        double T_max = 2.0/.197; 
        x1 = T_min;
        x2 = T_max;


        double copy(0.0) , prec = 1.e-6;
        int n = 0;
        int flag_1 = 0;
        do
        {
            mid = (x1 + x2) / 2.0;
            double e_mid = EquilibriumEnergyDensity(mid, params);
            double e1    = EquilibriumEnergyDensity(x1, params);


            if (abs(e_mid - e) < prec) 
                break;

            if ((e_mid - e) * (e1 - e) <= 0.0) 
                x2=mid;
            else
                x1=mid;

            n++;        
            if (n == 1) 
                copy = mid;

            if (n > 4)
            {
                if (abs(copy - mid) < prec)
                flag_1 = 1;	
                copy = mid;
            }
        }while (flag_1 != 1 && n <= 2000);

        return mid;	
    }
    // -------------------------------------



    AnisoHydroEvolution::TransportCoefficients AnisoHydroEvolution::CalculateTransportCoefficients(double e, double pt, double pl, double tau, SP& params)
    {
        double T  = InvertEnergyDensity(e, params);
        
        // Coefficients for relaxation times
        double c_tau_pi = params.c_tau_pi;
        double c_tau_Pi = params.c_tau_Pi;

        // Calculate shear pressure
        double pi = 2.0 / 3.0 * (pl - pt);
        double pi_bar = 3.0 * pi / (4.0 * e);
        // if (pi < 0) 
        // {
        //     Print_Error(std::cerr, "Shear cannot be negative.");
        //     exit(-4444);
        // }
        double xi = InvertShearToXi(e, pi);

        // Calculate transport coefficients
        double zetaBar_zL = e * R240(xi) / R200(xi) - 3.0 * pl;
        double zetaBar_zT = (e / 2.0) * R221(xi) / R200(xi) - pt;

        Print(std::cout, fmt::format("c_tau_pi = {:5f}, c_tau_Pi = {:.5f}, e = {:.5f}, pi_bar = {:.5f}, xi = {:.5f}, zeta_zL = {:.5f}, zeta_zT = {:.5f}", c_tau_pi, c_tau_Pi, e, pi_bar, xi, zetaBar_zL, zetaBar_zT));

        TransportCoefficients tc {c_tau_pi / T, c_tau_Pi / T, zetaBar_zT, zetaBar_zL};
        return tc;
    }
    // -------------------------------------



    double AnisoHydroEvolution::InvertShearToXi(double e, double pi)
    {
        if (pi == 0) return 0.0;

        double err = inf;
        double local_tol = 1e-6;
        double xi1 = -0.5;
        double xi2 = 0.5;
        double xin = 0.0;

        double pi_bar = 3.0 *  pi / (4.0 * e);
        double alpha = 1.0; // search speed;

        // function we want to find the root of
        std::function<double(double, double)> func = [this](double pi_bar, double xi) -> double
        {
            return pi_bar - 0.25 * (3.0 * R220(xi) / R200(xi) - 1);
        };

        while (err > local_tol)
        {
            Print(std::cout, fmt::format("xi1 = {:.5f}, xi2 = {:.5f}, xin = {:.5f}", xi1, xi2, xin));
            xin = xi2 - alpha * func(pi_bar, xi2) * (xi2 - xi1) / (func(pi_bar, xi2) - func(pi_bar, xi1));

            // Check for zero anisotropy case
            if (xin == 0) return 0;

            // calculate error and update variables
            err = std::fabs(xin - xi2) / std::fabs(xin);
            xi1 = xi2;
            xi2 = xin;
        }

        return xi2;
    }
    // -------------------------------------



    double AnisoHydroEvolution::R200(double xi)
    {
        if (xi == 0) return 1.0;
        else if (xi < 0) return 0.5 * (1.0 / (1.0 + xi) + std::atanh(std::sqrt(-xi)) / std::sqrt(-xi));
        else return 0.5 * (1.0 / (1.0 + xi) + std::atan(std::sqrt(xi)) / std::sqrt(xi));
    }
    // -------------------------------------



    double AnisoHydroEvolution::R220(double xi)
    {
        if (xi == 0) return 0.3333333;
        else if (xi < 0) 0.5 * (-1.0 / (1.0 + xi) + std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / xi;
        else return 0.5 * (-1.0 / (1.0 + xi) + std::atan(std::sqrt(xi)) / std::sqrt(xi)) / xi;
    }
    // -------------------------------------

    

    double AnisoHydroEvolution::R221(double xi)
    {
        if (xi == 0) return 4.0 / 30.0;
        else if (xi < 0) return 0.5 * (-3.0 + (3.0 + xi) * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / pow(xi, 2.0);
        else return 0.5 * (-3.0 + (3.0 + xi) * std::atan(std::sqrt(xi)) / std::sqrt(xi)) / pow(xi, 2.0);
    }
    // -------------------------------------



    double AnisoHydroEvolution::R240(double xi)
    {
        if (xi == 0) return 0.2;
        else if (xi < 0) return 0.5 * ((3.0 + 2.0 * xi) / (1.0 + xi) - 3.0 * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / pow(xi, 2.0);
        else return 0.5 * ((3.0 + 2.0 * xi) / (1.0 + xi) - 3.0 * std::atan(std::sqrt(xi)) / std::sqrt(xi)) / pow(xi, 2.0);
    }



    double AnisoHydroEvolution::dedt(double e, double pl, double tau)
    {
        return -(e + pl) / tau;
    }
    // -------------------------------------



    double AnisoHydroEvolution::dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
    {
        double tau_pi     = tc.tau_pi;
        double tau_Pi     = tc.tau_Pi;
        double zetaBar_zT = tc.zetaBar_zT;
        double pbar       = (pl + 2.0 * pt) / 3.0;
        return -(pbar - p) / tau_Pi +  (pl - pt) / (3.0 * tau_pi) + zetaBar_zT;
    }
    // -------------------------------------



    double AnisoHydroEvolution::dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
    {
        double tau_pi     = tc.tau_pi;
        double tau_Pi     = tc.tau_Pi;
        double zetaBar_zL = tc.zetaBar_zL;
        double pbar       = (pl + 2.0 * pt) / 3.0;
        return -(pbar - p) / tau_Pi -  (pl - pt) / (1.5 * tau_pi) + zetaBar_zL;
    }
}
