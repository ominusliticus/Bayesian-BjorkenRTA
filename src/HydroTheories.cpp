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
        return std::exp(exp * std::log(base));
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
        double dt = params.step_size;
        double m  = params.mass;

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
        
        // Initialize simulation
        T0         = 1.0 / params.D[0]; // Note that the temperature is already in units fm^{-1}
        double m   = params.mass;       // Note theat the mass in already in units fm^{-1}
        double z0  = m / T0;
        double e0  = 3.0 * pow(T0, 4.0) / (PI * PI) * (z0 * z0 * std::cyl_bessel_k(2, z0) / 2.0 + pow(z0, 3.0) * std::cyl_bessel_k(1, z0) / 6.0);   // Equilibrium energy density
        double p0  = z0 * z0 * pow(T0, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_kf(2.0, z0);                                                        // Equilibrium pressure

        // Note: all dynamic variables are declared as struct memebrs variables
        e1  = e0;
        pt1 = 3.0 * p0 / (2.0 + params.pl_pt_ratio);
        pl1 = 3.0 * p0 * params.pl_pt_ratio / (2.0 + params.pl_pt_ratio);

        aVars.Lambda = T0; aVars.alpha_T = 1.0; aVars.alpha_L = 2.0;

        FindAnisoVariables(e1, pt1, pl1, m, aVars);
        
        // Begin simulation 
        TransportCoefficients tc;
        double t;
        for (int n = 0; n < params.steps; n++)
        {
            t = t0 + n * dt;
            Print(e_plot , std::fixed, std::setprecision(decimal + 2), t, e1 );
            Print(pl_plot, std::fixed, std::setprecision(decimal + 2), t, pl1);
            Print(pt_plot, std::fixed, std::setprecision(decimal + 2), t, pt1);

            // RK4 with updating anisotropic variables
            // Note all dynamic variables are declared as member variables
            
            // First order
            FindAnisoVariables(e1, pt1, pl1, m, aVars);
            tc = CalculateTransportCoefficients(e1, pt1, pl1, params, aVars);
            de1  = dt *  dedt(e1, pl1, t);
            dpt2 = dt * dptdt(e1, pt1, pl1, t, tc);
            dpl2 = dt * dpldt(e1, pt1, pl1, t, tc);

            e2  = e1  + de1  / 2.0;
            pt2 = pt1 + dpt1 / 2.0;
            pl2 = pl1 + dpl1 / 2.0;

            // Second order
            FindAnisoVariables(e2, pt2, pl2, m, aVars);
            tc = CalculateTransportCoefficients(e2, pt2, pl2, params, aVars);
            de2  = dt *  dedt(e2, pl2, t);
            dpt2 = dt * dpldt(e2, pt2, pl2, t, tc);
            dpl2 = dt * dptdt(e2, pt2, pl2, t, tc);

            e3  = e2  + de2  / 2.0;
            pl3 = pl2 + dpt2 / 2.0;
            pt3 = pt2 + dpl2 / 2.0;

            // Third order
            FindAnisoVariables(e3, pt3, pl3, m, aVars);
            tc = CalculateTransportCoefficients(e3, pt3, pl3, params, aVars);
            de3  = dt *  dedt(e3, pl3, t);
            dpt3 = dt * dpldt(e3, pt3, pl3, t, tc);
            dpl3 = dt * dptdt(e3, pt3, pl3, t, tc);
            
            e4  = e3  + de3;
            pl4 = pl3 + dpt3;
            pt4 = pt3 + dpl3;

            // Fourth order
            FindAnisoVariables(e4, pt4, pl4, m, aVars);
            tc = CalculateTransportCoefficients(e4, pt4, pl4, params, aVars);
            de4  = dt *  dedt(e4, pl4, t);
            dpt4 = dt * dpldt(e4, pt4, pl4, t, tc);
            dpl4 = dt * dptdt(e4, pt4, pl4, t, tc);

            e1  += (de1  + 2.0 * de2  + 2.0 * de3  +  de4) / 6.0;
            pt1 += (dpt1 + 2.0 * dpt2 + 2.0 * dpt3 + dpt4) / 6.0;
            pl1 += (dpl1 + 2.0 * dpl2 + 2.0 * dpl3 + dpl4) / 6.0;

        } // End simulation loop
    }
    // -------------------------------------



    void AnisoHydroEvolution::ComputeF(double e, double pt, double pl, double mass, double (&X)[3], double F[3])
    {
        // F here is defined by Eq. (70) in arXiv:1803.01810 
        F[0] = IntegralI(2, 0, 0, 0, mass, X) - e;
        F[1] = IntegralI(2, 0, 1, 0, mass, X) - pt;
        F[2] = IntegralI(2, 2, 0, 0, mass, X) - pl;
    }
    // -------------------------------------

    

    void AnisoHydroEvolution::ComputeJ(double e, double pt, double pl, double mass, double (&X)[3], double F[3], double (*J)[3])
    {
        auto [Lambda, alpha_T, alpha_L] = X;
        double L2 = pow(Lambda, 2.0);
        double LaT3 = Lambda * pow(alpha_T, 3.0);
        double LaL3 = Lambda * pow(alpha_L, 3.0);

        // J is defined in Eq. (72) in arXiv:1803.01810 
        J[0][0] = IntegralJ(2, 0, 0, 0, mass, X) / L2;
        J[0][1] = 2.0 * IntegralJ(4, 0, 1, -1, mass, X) / LaT3;
        J[0][2] = IntegralJ(4, 2, 0, -1, mass, X) / LaL3;

        J[1][0] = IntegralJ(2, 2, 0, 1, mass, X) / L2;
        J[1][1] = 2.0 * IntegralJ(4, 2, 1, -1, mass, X) / LaT3;
        J[1][2] = IntegralJ(4, 4, 0, -1, mass, X) / LaL3;

        J[2][0] = IntegralJ(2, 0, 1, 1, mass, X) / L2;
        J[2][1] = 2.0 * IntegralJ(4, 0, 2, -1, mass, X) / LaT3;
        J[2][2] = IntegralJ(4, 2, 1, -1, mass, X) / LaL3;

    }
    // -------------------------------------



    double AnisoHydroEvolution::LineBackTrack(double e, double pt, double pl, double mass, const double Xcurrent[3], const double dX[3], double dX_magnitude, double g0, double F[3])
    {
        // Comment taken from Mike McNelis source code:
        //      This line backtracking algorithm is from the book Numerical Recipes in C

        //      initial data for g(delta_step) model:
        //      g0 = f(Xcurrent)                // f at Xcurrent
        //      f  = f(Xcurrent + dX)           // f at full newton step Xcurrent + dX
        //      gprime0 = - 2g0                 // descent derivative at Xcurrent


        double X[3];
        
        for (int i = 0; i < 3; i++)
        {
            X[i] = Xcurrent[i] + dX[i];     // Default Newton step
        }

        ComputeF(e, pt, pl, mass, X, F);    // Update F
        double f = (F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) / 2.0;
        double gprime0 = -2.0 * g0;

        double delta_step = 1.0;            // Default step parameter
        double alpha = 0.0001;              // descent rate

        double delta_root, delta_prev, f_prev;
        for(int n = 0; n < 20; n++)                     // line search iterations (max is 20)
        {
            if((delta_step * dX_magnitude) <= 1.e-4)                  // check if delta_step.|dX| within desired tolerance
            {
                return delta_step;
            }
            else if(f <= (g0  +  delta_step * alpha * gprime0))  // check for sufficient decrease in f
            {
                return delta_step;
            }
            else if(n == 0)                             // compute delta_step (start with quadratic model)
            {
                delta_root = - gprime0 / (2. * (f - g0 - gprime0));
            }
            else                                        // cubic model for subsequent iterations
            {
                double a = ((f  -  g0  -  delta_step * gprime0) / (delta_step * delta_step)  -  (f_prev  -  g0  -  delta_prev * gprime0) / (delta_prev * delta_prev)) / (delta_step - delta_prev);
                double b = (-delta_prev * (f  -  g0  -  delta_step * gprime0) / (delta_step * delta_step)  +  delta_step * (f_prev  -  g0  -  delta_prev * gprime0)  /  (delta_prev * delta_prev)) / (delta_step - delta_prev);

                if(a == 0)                              // quadratic solution to dg/dl = 0
                {
                    delta_root = - gprime0 / (2. * b);
                }
                else
                {
                    double z = b * b  -  3. * a * gprime0;

                    if(z < 0)
                    {
                        delta_root = 0.5 * delta_step;
                    }
                    else if(b <= 0)
                    {
                        delta_root = (-b + std::sqrt(z)) / (3. * a);
                    }
                    else
                    {
                        delta_root = - gprime0 / (b + std::sqrt(z));   // what does this mean?
                    }
                }

                delta_root = std::fmin(delta_root, 0.5 * delta_step);
            }

            delta_prev = delta_step;                                  // store current values for the next iteration
            f_prev = f;

            delta_step = std::fmax(delta_root, 0.1 * delta_step);                   // update delta_step and f

            for(int i = 0; i < 3; i++)
            {
                X[i] = Xcurrent[i]  +  delta_step * dX[i];
            }

            ComputeF(e, pt, pl, mass, X, F);

            f = (F[0] * F[0]  +  F[1] * F[1]  +  F[2] * F[2]) / 2.;
        }

        return delta_step;
    }
    // -------------------------------------



    void AnisoHydroEvolution::FindAnisoVariables(double e, double pt, double pl, double mass, AnisoVariables aVars)
    {
        double X[3] {aVars.Lambda, aVars.alpha_T, aVars.alpha_L};  // Current solution
        double dX[3]   {};  // Iteration step
        double F[3]    {};  // Evaluation of function we want to minimize, F(X)
        double J[3][3] {};  // Matrix from analytically inverting (E, PT, PL) -> (Lambda, alpha_T, alpha_L)

        // Scaled minimum step length allowed in line search
        double step_max = 100 * std::fmax(std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]), 3.0);

        // Let's minimize F(X)
        ComputeF(e, pt, pl, mass, X, F);
        gsl_vector * x = gsl_vector_alloc(3);							// holds dX
        gsl_permutation * p = gsl_permutation_alloc(3);
        
        double tol = 1.e-6;
        for (int n = 0; n < 100; n++)                                   // Is 100 enough?
        {
            ComputeJ(e, pt, pl, mass, X, F, J);
            double f = (F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) / 2.;	// f = F(X).F(X) / 2
            double J_gsl[9] = {J[0][0], J[0][1], J[0][2],
                               J[1][0], J[1][1], J[1][2],
                               J[2][0], J[2][1], J[2][2]};				// Jacobian matrix in gsl format

            for (int i; i < 3; i++) F[i] *= -1.0;                       // Flig sign of F

            // solve matrix equations J.dX = -F
            int s; 														
            gsl_matrix_view A = gsl_matrix_view_array(J_gsl, 3, 3);
            gsl_vector_view b = gsl_vector_view_array(F, 3);
            gsl_linalg_LU_decomp(&A.matrix, p, &s);
            gsl_linalg_LU_solve(&A.matrix, p, &b.vector, x);
            for (int i = 0; i < 3; i++) dX[i] = gsl_vector_get(x, i);
            
            double dX_magnitude = sqrt(dX[0] * dX[0]  +  dX[1] * dX[1]  +  dX[2] * dX[2]);
            if (dX_magnitude > step_max)                                 // Rescale if dX to large
            {
                for (int i = 0; i < 3; i++) dX[i] *= step_max / dX_magnitude;
                dX_magnitude = step_max;
            }

            double delta_step = LineBackTrack(e, pt, pl, mass, X, dX, dX_magnitude, f, F);
            for (int i = 0; i < 3; i++) X[i] += delta_step * dX[i];

            // LineBackTrack also modifies F(X)
            double F_magnitude = sqrt(F[0] * F[0]  +  F[1] * F[1]  +  F[2] * F[2]);
            dX_magnitude *= delta_step;

            // Check if any quantities have gone negative. If yes, then break.
            if (X[0] < 0 || X[1] < 0 || X[3]) 
            {                
                gsl_permutation_free(p);
       		    gsl_vector_free(x);
                break;
            }

            // Check for convergence of F(X)
            if (dX_magnitude < tol && F_magnitude < tol)
            {
                aVars.Lambda  = X[0];
                aVars.alpha_T = X[1];
                aVars.alpha_L = X[3];

                gsl_permutation_free(p);
                gsl_vector_free(x);
                break;
            }
        } // End Newton-Coats loop
    }
    // -------------------------------------



    double AnisoHydroEvolution::IntegralI(int n, int q, int r, int s, double mass, double (&X)[3])
    {
        return GausQuad([this](double pbar, int n, int q, int r, int s, double mass, double (&X)[3])
        {
            return IntegralIAux(n, q, r, s, mass, pbar, X);
        }, 0, inf, tol, max_depth, n, q, r, s, mass, X);
    }
    // -------------------------------------



    double AnisoHydroEvolution::IntegralIAux(int n, int q, int r, int s, double mass, double pbar, double (&X)[3])
    {
        // Get anisotropic variables using structured binding
        auto [Lambda, alpha_T, alpha_L] = X;
        double mbar      = mass / Lambda;
        double prefactor = pow(alpha_T, 2 * q + 2) * pow(alpha_L, r + 1) * pow(Lambda, n + s + 2) / (4 * PI * PI * doubleFactorial(2 * q));
        double pbar_ns1  = pow(pbar, n + 2 + 1);
        double Rnrq      = IntegrandR(n, r, q, mass, pbar, X);
        double feq       = std::exp(-std::sqrt(pbar * pbar + mbar * mbar));
        return prefactor * pbar_ns1 * Rnrq * feq;
    }
    // -------------------------------------



    double AnisoHydroEvolution::IntegralJ(int n, int q, int r, int s, double mass, double (&X)[3])
    {
        return GausQuad([this](double pbar, int n, int q, int r, int s, double mass, double (&X)[3])
        {
            return IntegralJAux(n, q, r, s, mass, pbar, X);
        }, 0, inf, tol, max_depth, n, q, r, s, mass, X);
    }
    // -------------------------------------



    double AnisoHydroEvolution::IntegralJAux(int n, int q, int r, int s, double mass, double pbar, double (&X)[3])
    {
        // Get anisotropic variables using structured binding
        auto [Lambda, alpha_T, alpha_L] = X;
        double mbar      = mass / Lambda;
        double prefactor = pow(alpha_T, 2 * q + 2) * pow(alpha_L, r + 1) * pow(Lambda, n + s + 2) / (4 * PI * PI * doubleFactorial(2 * q));
        double pbar_ns1  = pow(pbar, n + 2 + 1);
        double Rnrq      = IntegrandR(n, r, q, mass, pbar, X);
        double feq       = std::exp(-std::sqrt(pbar * pbar + mbar * mbar));
        return prefactor * pbar_ns1 * Rnrq * feq * (1 - feq);
    }
    // -------------------------------------



    double AnisoHydroEvolution::IntegrandR(int n, int q, int r, double mass, double pbar, double (&X)[3])
    {
        auto [Lambda, alpha_T, alpha_L] = X;
        double mbar = mass / Lambda;
        double w    = std::sqrt(alpha_L * alpha_L + pow(mbar / pbar, 2.0));
        double z    = (alpha_T * alpha_T - alpha_L * alpha_L) / (w * w);
        return GausQuad([this](double theta, int n, int q, int r, double z, double w)
        {
            double w_nr2q1      = pow(w, n - r - 2 * q - 1);
            double sintheta_2q1 = pow(std::sin(theta), 2 * q + 1);
            double costheta_r   = pow(std::cos(theta), r);
            double val          = pow(1 + z * pow(std::sin(theta), 2.0), (n - r - 2 * q - 1) / 2.0);
            return w_nr2q1 * sintheta_2q1 * costheta_r * val;
        }, 0, PI, tol, max_depth, n, q, r, z, w);
    }
    // -------------------------------------



    double AnisoHydroEvolution::EquilibriumEnergyDensity(double temp, SP& params)
    {
        double z = params.mass / temp;
        return 3.0 * pow(temp, 4.0) / (PI * PI) * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
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



    AnisoHydroEvolution::TransportCoefficients& AnisoHydroEvolution::CalculateTransportCoefficients(double e, double pt, double pl, SP& params, AnisoVariables aVars)
    {
        double T  = InvertEnergyDensity(e, params);
        double z  = params.mass / T;

        // terms necessary for calculating beta_pi and beta_Pi
        // c.f. Eq. (60) in arXiv:1612.07329
        double K5  = std::cyl_bessel_k(5, z);
        double K3  = std::cyl_bessel_k(3, z);
        double K2  = std::cyl_bessel_k(2, z);
        double K1  = std::cyl_bessel_k(1, z);
        double Ki1 = GausQuad([](double th, double z){ return std::exp(- z * std::cosh(th)) / std::cosh(th); }, 0, inf, tol, max_depth, z); 
        
        double peq  = z * z * std::pow(T, 4.0) / (2.0 * PI * PI) * K2;                          // Equilibrium pressure
        double cs2  = (e + peq) / (3.0 * e + (3.0 + z * z) * peq);                              // Sound speed squared
        double s    = pow(T, 4.0) / (2.0 * PI * PI) * (4.0 * z * z * K2 + pow(z, 3.0) * K1);    // Equilibrium entropy density
 
        double beta_pi = pow(T * z, 5.0) / (30.0 * PI * PI) * ((K5 - 7.0 * K3 + 22.0 * K1) / 16.0 - Ki1);
        double beta_Pi = 5.0 * beta_pi / 3.0 - cs2 * (e + peq);

        // c.f. Eqs. (78) - (80)
        double eta_s_min   = params.eta_s_min;
        double eta_s_slope = params.eta_s_slope;
        double zeta_s_norm = params.zeta_s_norm;
        double Tc          = params.T_c;
        
        double eta_s;
        if (T > Tc) eta_s = eta_s_min + eta_s_slope * (T - Tc);
        else eta_s = eta_s_min;

        double zeta_s = zeta_s_norm;
        double x = T / Tc;
        if (x < 0.995) zeta_s *= 0.03 + 0.9 * std::exp((x - 1.0) / 0.0025) + 0.22 * std::exp((x - 1.0) / 0.022);
        else if (x < 1.05) zeta_s *= -13.45 + 27.55 * x - 13.77 * x * x;
        else zeta_s *= 0.001 + 0.9 * std::exp((x - 1.0) / 0.025) + 0.25 * std::exp((x - 1.0) / 0.13);


        // c.f. Eqs. (C1a) amd (C2a) of arXiv:1803.01810
        double X[3] {aVars.Lambda, aVars.alpha_T, aVars.alpha_L};
        double zetaBar_zL = IntegralI(2, 4, 0, 0, params.mass, X) - 3.0 * pl;
        double zetaBar_zT = IntegralI(2, 2, 0, 0, params.mass, X) - pt;

        TransportCoefficients tc {eta_s * s / beta_pi, zeta_s * s / beta_Pi, zetaBar_zT, zetaBar_zL};
        return tc;
    }
    // -------------------------------------



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
