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

    double DoubleFactorial(int k)
    {
        if (k <= 1)
            return 1.0;
        double result = (double)k * DoubleFactorial(k - 2);
        return result;
    }
    // -------------------------------------

    double Gamma(double z)
    {
        return GausQuad([](double x, double z)
        {
            return std::exp(-x) * pow(x, z - 1);
        }, 0 , inf, tol , max_depth, z);
    }

    ///////////////////////////////////
    // Viscous struct implementation //
    ///////////////////////////////////
    void ViscousHydroEvolution::RunHydroSimulation(SP& params, theory theo)
    {
        double t0 = params.tau_0;
        double dt = params.step_size;

        // for setprecision of t output
        int decimal = - (int) std::log10(dt);

        // Opening output files
        std::fstream e_plot, pi_plot, Pi_plot;
        switch (theo)
        {
            case theory::CE:
                Print(std::cout, "Calculting viscous hydro in Chapman-Enskog approximation");
                e_plot  = std::fstream("output/CE_hydro/e.dat", std::ios::out);
                pi_plot = std::fstream("output/CE_hydro/shear.dat", std::ios::out);
                Pi_plot = std::fstream("output/CE_hydro/bulk.dat", std::ios::out);
                break;
            case theory::DNMR:
                Print(std::cout, "Calculting viscous hydro in 14-moment approximation");
                e_plot  = std::fstream("output/DNMR_hydro/e.dat", std::ios::out);
                pi_plot = std::fstream("output/DNMR_hydro/shear.dat", std::ios::out);
                Pi_plot = std::fstream("output/DNMR_hydro/bulk.dat", std::ios::out);
                break;
        }
        if (!e_plot.is_open() && !pi_plot.is_open() && !Pi_plot.is_open())
        {
            Print_Error(std::cerr, "ViscousHydroEvolution::RunHydroSimulation: Failed to open output files.");
            switch (theo)
            {
                case theory::CE:
                    Print_Error(std::cerr, "Pleae make sure the folder ./output/CE_hydro/ exists.");
                    break;
                case theory::DNMR:
                    Print_Error(std::cerr, "Pleae make sure the folder ./output/DNMR_hydro/ exists.");
                    break;
            }
            exit(-3333);
        }
        else
        {
            e_plot  << std::fixed << std::setprecision(decimal + 10);
            pi_plot << std::fixed << std::setprecision(decimal + 10);
            Pi_plot << std::fixed << std::setprecision(decimal + 10);
        }
        
        // Initialize simulation
        double T0  = 0.5 / 0.197; //1.0 / params.D[0]; // Note that the temperature is already in units fm^{-1}
        double m   = params.mass;       // Note that the mass in already in units fm^{-1}
        double z0  = m / T0;
        double e0;                      // Equilibrium energy density
        if (z0 == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
        else e0 = 3.0 * pow(T0, 4.0) / (PI * PI) * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

        // Thermal pressure necessary for calculating bulk pressure and inverting pi to xi
        auto ThermalPressure = [this](double e, SP& params) -> double
        {
            double T = InvertEnergyDensity(e, params);
            double z = params.mass / T;
            if (z == 0) return pow(T, 4.0) / (PI * PI);
            else return z * z * pow(T, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_k(2, z);
        };

        // Note: all dynamic variables are declared as struct memebrs variables
        e1  = e0;
        if (m == 0)
        {
            double pt0 = e0 / 4.0;
            double pl0 = e0 / 2.0;
            pi1  = 2.0 * (pt0 - pl0) / 3.0;
            Pi1 = 0.0;
        }
        else
        {
            pi1 = 2.0 * (params.pt0 - params.pl0) / 3.0;
            Pi1 = (params.pl0 + 2.0 * params.pt0) / 3.0 - ThermalPressure(e0, params);
        }
        
        // Begin simulation 
        TransportCoefficients tc;
        double t;
        for (int n = 0; n < params.steps; n++)
        {
            t = t0 + n * dt;
            
            p1 = ThermalPressure(e1, params);
            Print(e_plot,  t, e1, p1);
            Print(pi_plot, t, pi1);
            Print(Pi_plot, t, Pi1);

            // Invert energy density to compute thermal pressure

            // RK4 with updating anisotropic variables
            // Note all dynamic variables are declared as member variables
            // fmt::print("e1 = {}, pi1 = {}, Pi1 = {}\n", e1, pi1, Pi1);
            
            // First order
            tc = CalculateTransportCoefficients(e1, pi1, Pi1, params, theo);
            de1  = dt *  dedt(e1, p1, pi1, Pi1, t);
            dpi1 = dt * dpidt(pi1, Pi1, t, tc);
            dPi1 = dt * dPidt(pi1, Pi1, t, tc);

            e2  = e1  + de1  / 2.0;
            pi2 = pi1 + dpi1 / 2.0;
            Pi2 = Pi1 + dPi1 / 2.0;

            // Second order
            p2 = ThermalPressure(e2, params);
            tc = CalculateTransportCoefficients(e2, pi2, Pi2, params, theo);
            de2  = dt *  dedt(e2, p2, pi2, Pi2, t  + dt / 2.0);
            dpi2 = dt * dpidt(pi2, Pi2, t + dt / 2.0, tc);
            dPi2 = dt * dPidt(pi2, Pi2, t + dt / 2.0, tc);

            e3  = e1  + de2  / 2.0;
            pi3 = pi1 + dpi2 / 2.0;
            Pi3 = Pi1 + dPi2 / 2.0;

            // Third order
            p3 = ThermalPressure(e3, params);
            tc = CalculateTransportCoefficients(e3, pi3, Pi3, params, theo);
            de3  = dt *  dedt(e3, p3, pi3, Pi3, t + dt / 2.0);
            dpi3 = dt * dpidt(pi3, Pi3, t + dt / 2.0, tc);
            dPi3 = dt * dPidt(pi3, Pi3, t + dt / 2.0, tc);
            
            e4  = e1  + de3;
            pi4 = pi1 + dpi3;
            Pi4 = Pi1 + dPi3;

            // Fourth order
            p4 = ThermalPressure(e4, params);
            tc = CalculateTransportCoefficients(e4, pi4, Pi4, params, theo);
            de4  = dt *  dedt(e4, p4, pi4, Pi4, t + dt);
            dpi4 = dt * dpidt(pi4, Pi4, t + dt, tc);
            dPi4 = dt * dPidt(pi4, Pi4, t + dt, tc);

            e1  += (de1  + 2.0 * de2  + 2.0 * de3  +  de4) / 6.0;
            pi1 += (dpi1 + 2.0 * dpi2 + 2.0 * dpi3 + dpi4) / 6.0;
            Pi1 += (dPi1 + 2.0 * dPi2 + 2.0 * dPi3 + dPi4) / 6.0;
        } // End simulation loop
    }
    // -------------------------------------



    double ViscousHydroEvolution::EquilibriumEnergyDensity(double temp, SP& params)
    {
        double z = params.mass / temp;
        if (z == 0) return 3.0 * pow(temp, 4.0) / (PI * PI);
        else return 3.0 * pow(temp, 4.0) / (PI * PI) * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
    }
    // -------------------------------------



    double ViscousHydroEvolution::InvertEnergyDensity(double e, SP& params)
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



    ViscousHydroEvolution::TransportCoefficients ViscousHydroEvolution::CalculateTransportCoefficients(double e, double pi, double Pi, SP& params, theory theo)
    {
        // invert energy density to temperature
        double T = InvertEnergyDensity(e, params);
        double m = params.mass;
        double z = m / T;
        double beta = 1.0 / T;


        switch (theo)
        {
            case theory::CE:
            {
                // Thermodynamic integrals, c.f. Eqs. (45) - (48) in arXiv:1407.7231
                double I3_63, I1_42, I3_42, I2_31, I0_31, I0_30;
                if (z == 0)
                {
                    I3_63 = -4.0 * pow(T, 5.0) / (35.0 * PI * PI);
                    I1_42 = 4.0 * pow(T, 5.0) / (5.0 * PI * PI);
                    I3_42 = pow(T, 3.0) / (15.0 * PI * PI);
                    I2_31 = -pow(T, 3.0) / (3.0 * PI * PI);
                    I0_31 = -4.0 * e * T / 3.0;
                    I0_30 = 4.0 * e * T;
                }
                else
                {
                    double K5 = std::cyl_bessel_k(5, z);
                    double K3 = std::cyl_bessel_k(3, z);
                    double K2 = std::cyl_bessel_k(2, z);
                    double K1 = std::cyl_bessel_k(1, z);

                    double Ki3 = GausQuad([](double th, double z){ return std::exp(- z * std::cosh(th)) * pow(std::cosh(th), -3); }, 0, inf, tol, max_depth, z);
                    double Ki1 = GausQuad([](double th, double z){ return std::exp(- z * std::cosh(th)) * pow(std::cosh(th), -1); }, 0, inf, tol, max_depth, z);

                    I3_63 = -pow(T * z, 5.0) / (210 * PI * PI) * ((K5 - 11.0 * K3 + 58.0 * K1) / 16.0 - 4.0 * Ki1 + Ki3);
                    I1_42 = pow(T * z, 5.0) / (30.0 * PI * PI) * ((K5 - 7.0 * K3 + 22.0 * K1) / 16.0 - Ki1);
                    I3_42 = pow(T * z, 3.0) / (30.0 * PI * PI) * ((K3 - 9.0 * K1) * 0.25 + 3.0 * Ki1 - Ki3);
                    I2_31 = -pow(T * z, 3.0) / (6.0 * PI * PI) * ((K3 - 5.0 * K1) * 0.25 + Ki1);
                    I0_31 = -(e + z * z * pow(T, 4.0) / (2.0 * PI * PI) * K2) * T;
                    I0_30 = (3.0 * e + (3.0 + z * z) * z * z * pow(T, 4.0) / (2.0 * PI * PI) * K2) * T;
                }

                // sound speed sqrd
                double cs2 = -I0_31 / I0_30;

                // calculatiung transport coefficients
                // Eqs. (25) - (26) in arXiv:1407.7231
                double beta_pi = beta * I1_42;
                double beta_Pi = 5.0 * beta_pi / 3.0 + beta * I0_31 * cs2;
                // double s = - beta * beta * I0_31;                             // thermal entropy density

                // TO DO: should relaxation time always be the same in Bjorken flow?
                double tau_pi = 5.0 * params.eta_s / T;
                double tau_Pi = tau_pi;

                // Eqs. (35) - (40) arXiv:1407:7231
                double chi         = beta * ((1.0 - 3.0 * cs2) * (I1_42 + I0_31) - m * m * (I3_42 + I2_31)) / beta_Pi;
                double delta_PiPi  = - 5.0 * chi / 9.0 - cs2;
                double lambda_Pipi = beta * (7.0 * I3_63 + 2.0 * I1_42) / (3.0 * beta_pi) - cs2; 
                double tau_pipi    = 2.0 + 4.0 * beta * I3_63 / beta_pi;
                double delta_pipi  = 5.0 / 3.0 + 7.0 * beta * I3_63 / (3.0 * beta_pi);
                double lambda_piPi = - 2.0 * chi / 3.0;

                double check1 = std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2);
                double check2 = std::fabs(lambda_Pipi - delta_pipi + 1 + cs2);
                double check3 = std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0);
                
                TransportCoefficients tc {tau_pi, beta_pi, tau_Pi, beta_Pi, delta_pipi, delta_PiPi, lambda_piPi, lambda_Pipi, tau_pipi};
                double local_tol = 5 * tol;
                if (check1 < local_tol && check2 < local_tol && check3 < local_tol) return tc;
                else 
                {
                    Print_Error(std::cerr, "ViscousHydroEvolution::CalculateTransportCoefficients: transport coefficients did not satisfy relations."); 
                    Print_Error(std::cerr, fmt::format("std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2)      = {}", check1));
                    Print_Error(std::cerr, fmt::format("std::fabs(lambda_Pipi - delta_pipi + 1 + cs2)              = {}", check2));
                    Print_Error(std::cerr, fmt::format("std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0) = {}", check3));
                    exit(-5555);
                }
                break;
            }

            case theory::DNMR: // Reference Appendix E in arXiv:1803.01810
            {
                auto IntegralI = [](int n, int q, double z, double T)
                {
                    if (z == 0) 
                    {
                        return pow(T, n + 2) * Gamma(n + 2) / (2.0 * PI * PI * DoubleFactorial(2 * q + 1));
                    }
                    else 
                    {
                        return GausQuad([](double x, double z, double T, int n, int q)
                        {
                            return pow(T * z, n + 2) * pow(x, n - 2 * q) * pow(x * x - 1.0, (2 * q + 1) * 0.5) * std::exp(-z * x)  / (2.0 * PI * PI * DoubleFactorial(2 * q + 1));; 
                        }, 1, inf, tol, max_depth, z, T, n, q);
                    }
                };
                
                double I00 = IntegralI(0, 0, z, T);
                double I01 = IntegralI(0, 1, z, T);
                double I22 = IntegralI(2, 2, z, T);
                double I30 = IntegralI(3, 0, z, T);
                double I31 = IntegralI(3, 1, z, T);
                double I32 = IntegralI(3, 2, z, T);
                double I40 = IntegralI(4, 0, z, T);
                double I41 = IntegralI(4, 1, z, T);
                double I42 = IntegralI(4, 2, z, T);

                // sound speed sqrd
                double cs2     = I31 / I30;
                double cBar_e  = -I41 / (5.0 * I40 * I42 / 3.0 - I41 * I41);
                double cBar_Pi = I40 / (5.0 * I40 * I42 / 3.0 - I41 * I41);
                if (z == 0)
                {
                    cBar_e = 0;
                    cBar_Pi = 0;
                }
                double cBar_pi = 1.0 / I42;

                double beta_pi = beta * I32;
                double beta_Pi = 5.0 * beta_pi / 3.0 - beta * I31 * cs2;
                // double s = beta * beta * I31;                             // thermal entropy density

                double tau_pi = 5.0 * params.eta_s / T;
                double tau_Pi = tau_pi;

                double delta_PiPi  = 1.0 - cs2 - pow(m, 4.0) * (cBar_e * I00 + cBar_Pi * I01) / 9.0;
                double lambda_Pipi = (1.0  + cBar_pi * m * m * I22) / 3.0 - cs2; 
                double tau_pipi    = (10.0 + 4.0 * cBar_pi * m * m * I22) / 7.0;
                double delta_pipi  = (4.0 + cBar_pi * m * m * I22) / 3.0;;
                double lambda_piPi = 6.0 / 5.0 - 2.0 * pow(m, 4.0) * (cBar_e * I00 + cBar_Pi * I01) / 15;

                double check1 = std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2);
                double check2 = std::fabs(lambda_Pipi - delta_pipi + 1 + cs2);
                double check3 = std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0);
                
                TransportCoefficients tc {tau_pi, beta_pi, tau_Pi, beta_Pi, delta_pipi, delta_PiPi, lambda_piPi, lambda_Pipi, tau_pipi};
                double local_tol = 5 * tol;
                if (check1 < local_tol && check2 < local_tol && check3 < local_tol) return tc;
                else 
                {
                    Print_Error(std::cerr, "ViscousHydroEvolution::CalculateTransportCoefficients: transport coefficients did not satisfy relations.");
                    Print_Error(std::cerr, fmt::format("std::fabs(delta_PiPi - 5.0 * lambda_piPi / 6.0 + cs2)      = {}", check1));
                    Print_Error(std::cerr, fmt::format("std::fabs(lambda_Pipi - delta_pipi + 1 + cs2)              = {}", check2));
                    Print_Error(std::cerr, fmt::format("std::fabs(tau_pipi - 6.0 * (2.0 * delta_pipi - 1.0) / 7.0) = {}", check3)); 
                    exit(-5555);
                }
                break;
            }
        } // End switch(theo)
        
        return {};
    }
    // -------------------------------------



    double ViscousHydroEvolution::dedt(double e, double p, double pi, double Pi, double tau)
    {
        return - (e + p + Pi - pi) / tau;
    }
    // -------------------------------------



    double ViscousHydroEvolution::dpidt(double pi, double Pi, double tau, TransportCoefficients& tc)
    {
        return -pi / tc.tau_pi + 4.0 * tc.beta_pi / (3.0 * tau) - (tc.tau_pipi / 3.0 + tc.delta_pipi) * pi / tau + 2.0 * tc.lambda_piPi * Pi / (3.0 * tau);
    }
    // -------------------------------------



    double ViscousHydroEvolution::dPidt(double pi, double Pi, double tau, TransportCoefficients& tc)
    {
        return - Pi / tc.tau_Pi - tc.beta_Pi / tau - tc.delta_PiPi * Pi / tau + tc.lambda_Pipi * pi / tau;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////
    // Anisotropic struct implementation //
    ///////////////////////////////////////
    void AnisoHydroEvolution::RunHydroSimulation(SP& params)
    {
        Print(std::cout, "Calculating anistropic hydrodynamic evolution");
        double t0 = params.tau_0;
        double dt = params.step_size;

        // for setprecision of t output
        int decimal = - (int) std::log10(dt);

        // Opening output files
        std::fstream e_plot("output/aniso_hydro/e.dat", std::ios::out);
        std::fstream pt_plot("output/aniso_hydro/pl.dat", std::ios::out);
        std::fstream pl_plot("output/aniso_hydro/pt.dat", std::ios::out);
        if (!e_plot.is_open() && !pt_plot.is_open() && !pl_plot.is_open())
        {
            Print_Error(std::cerr, "AnisoHydroEvolution::RunHydroSimulation: Failed to open output files.");
            Print_Error(std::cerr, "Pleae make sure the folder ./output/aniso_hydro/ exists.");
            exit(-3333);
        }
        else
        {
            e_plot  << std::fixed << std::setprecision(decimal + 10);
            pt_plot << std::fixed << std::setprecision(decimal + 10);
            pl_plot << std::fixed << std::setprecision(decimal + 10);
        }
        
        // Initialize simulation
        T0         = 0.5 / 0.197; //1.0 / params.D[0]; // Note that the temperature is already in units fm^{-1}
        double m   = params.mass;       // Note that the mass in already in units fm^{-1}
        double z0  = m / T0;
        double e0;                      // Equilibrium energy density
        if (z0 == 0) e0 = 3.0 * pow(T0, 4.0) / (PI * PI);
        else e0 = 3.0 * pow(T0, 4.0) / (PI * PI) * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);

        // Thermal pressure necessary for calculating bulk pressure and inverting pi to xi
        auto ThermalPressure = [this](double e, double pt, double pl, SP& params) -> double
        {
            double T = InvertEnergyDensity(e, params);
            double z = params.mass / T;
            if (z == 0) return (pl + 2.0 * pt) / 3.0;
            else return z * z * pow(T, 4.0) / (2.0 *PI * PI) * std::cyl_bessel_k(2, z);
        };

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
        for (int n = 0; n < params.steps; n++)
        {
            t = t0 + n * dt;
            p1 = ThermalPressure(e1, pt1, pl1, params);
            Print(e_plot,  t, e1, p1);
            Print(pt_plot, t, pl1);
            Print(pl_plot, t, pt1);

            // Invert energy density to compute thermal pressure

            // RK4 with updating anisotropic variables
            // Note all dynamic variables are declared as member variables
            
            // First order
            tc = CalculateTransportCoefficients(e1, p1, pt1, pl1, params);
            de1  = dt *  dedt(e1, pl1, t);
            dpt1 = dt * dptdt(p1, pt1, pl1, t, tc);
            dpl1 = dt * dpldt(p1, pt1, pl1, t, tc);

            e2  = e1  + de1  / 2.0;
            pt2 = pt1 + dpt1 / 2.0;
            pl2 = pl1 + dpl1 / 2.0;

            // Second order
            p2 = ThermalPressure(e2, pt2, pl2, params);
            tc = CalculateTransportCoefficients(e2, p2, pt2, pl2, params);
            de2  = dt *  dedt(e2, pl2, t  + dt / 2.0);
            dpt2 = dt * dptdt(p2, pt2, pl2, t + dt / 2.0, tc);
            dpl2 = dt * dpldt(p2, pt2, pl2, t + dt / 2.0, tc);

            e3  = e1  + de2  / 2.0;
            pt3 = pt1 + dpt2 / 2.0;
            pl3 = pl1 + dpl2 / 2.0;

            // Third order
            p3 = ThermalPressure(e3, pt3, pl3, params);
            tc = CalculateTransportCoefficients(e3, p3, pt3, pl3, params);
            de3  = dt *  dedt(e3, pl3, t + dt / 2.0);
            dpt3 = dt * dptdt(p3, pt3, pl3, t + dt / 2.0, tc);
            dpl3 = dt * dpldt(p3, pt3, pl3, t + dt / 2.0, tc);
            
            e4  = e1  + de3;
            pt4 = pt1 + dpt3;
            pl4 = pl1 + dpl3;

            // Fourth order
            p4 = ThermalPressure(e4, pt4, pl4, params);
            tc = CalculateTransportCoefficients(e4, p4, pt4, pl4, params);
            de4  = dt *  dedt(e4, pl4, t + dt);
            dpt4 = dt * dptdt(p4, pt4, pl4, t + dt, tc);
            dpl4 = dt * dpldt(p4, pt4, pl4, t + dt, tc);

            e1  += (de1  + 2.0 * de2  + 2.0 * de3  +  de4) / 6.0;
            pt1 += (dpt1 + 2.0 * dpt2 + 2.0 * dpt3 + dpt4) / 6.0;
            pl1 += (dpl1 + 2.0 * dpl2 + 2.0 * dpl3 + dpl4) / 6.0;

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



    AnisoHydroEvolution::TransportCoefficients AnisoHydroEvolution::CalculateTransportCoefficients(double e, double p, double pt, double pl, SP& params)
    {
        double T = InvertEnergyDensity(e, params);
        
        // Coefficients for relaxation times
        // TO DO: should the relaxation times always be equal in Bjorken flow?
        double tau_pi = 5.0 * params.eta_s / T;
        double tau_Pi = tau_pi;

        // Calculate shear pressure
        double pi = 2.0 / 3.0 * (pt - pl);
        double xi = InvertShearToXi(e, p, pi);

        // Calculate transport coefficients
        double zetaBar_zL = e * R240(xi) / R200(xi) - 3.0 * pl;
        double zetaBar_zT = (e / 2.0) * R221(xi) / R200(xi) - pt;
        if (params.mass == 0) zetaBar_zL = -(e + pl + 2.0 * zetaBar_zT);

        TransportCoefficients tc {tau_pi, tau_Pi, zetaBar_zT, zetaBar_zL};
        return tc;
    }
    // -------------------------------------



    double AnisoHydroEvolution::InvertShearToXi(double e, double p, double pi)
    {
        if (pi == 0) return 0.0;

        double err = inf;
        double local_tol = 1e-6;
        double xi1 = -0.5;
        double xi2 = 0.5;
        double xin = 0.0;

        double piBar = pi / (e + p);
        double alpha = 1.0e-1; // search speed;

        // function we want to find the root of
        auto func = [this](double piBar, double xi) -> double
        {
            // if (piBar < -0.5)
            // {
                
            // }
            return piBar - 0.6666667 * (0.5 * R201(xi) - R220(xi)) / R200(xi);
        };

        while (err > local_tol)
        {
            xin = xi2 - alpha * func(piBar, xi2) * (xi2 - xi1) / (func(piBar, xi2) - func(piBar, xi1));

            // Check for zero anisotropy case
            if (xin == 0) return 0;

            // calculate error and update variables
            err = std::fabs(xin - xi2) / std::fabs(xin);
            xi1 = xi2;
            xi2 = xin;
        }

        return xin;
        
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
        else if (xi < 0) return 0.5 * (-1.0 / (1.0 + xi) + std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / xi;
        else return 0.5 * (-1.0 / (1.0 + xi) + std::atan(std::sqrt(xi)) / std::sqrt(xi)) / xi;
    }
    // -------------------------------------



    double AnisoHydroEvolution::R201(double xi)
    {
        if (xi == 0) return 0.5;
        else if (xi < 0) return 0.5 * (1.0 - (1.0 - xi) * std::atanh(std::sqrt(-xi)) / std::sqrt(-xi)) / xi; 
        else return 0.5 * (1.0 - (1.0 - xi) * std::atan(std::sqrt(xi)) / std::sqrt(xi)) / xi;
    }

    

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
        return -(pbar - p) / tau_Pi + (pl - pt) / (3.0 * tau_pi) + zetaBar_zT / tau;
    }
    // -------------------------------------



    double AnisoHydroEvolution::dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc)
    {
        double tau_pi     = tc.tau_pi;
        double tau_Pi     = tc.tau_Pi;
        double zetaBar_zL = tc.zetaBar_zL;
        double pbar       = (pl + 2.0 * pt) / 3.0;
        return -(pbar - p) / tau_Pi - (pl - pt) / (1.5 * tau_pi) + zetaBar_zL / tau;
    }
}
