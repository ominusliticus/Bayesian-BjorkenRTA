#include "../include/ExactSolution.hpp"
#include "../include/Integration.hpp"

#include <iomanip>
#include <omp.h>



// Note: that the integration functions are written like this to make sure the 
// integration routine GausQuad is aware of th calss instance and can see that functions

namespace exact{
    
    // const double tol = eps;
    // const int max_depth = 5;
    // const int max_depth2 = 5;
    const double tol     = eps; //eps;
    const int max_depth  = 0;
    const int max_depth2 = 1;
    
    double EquilibriumEnergyDensity(double temp, SP& params)
    {
        return GausQuad(EquilibriumEnergyDensityAux, 0.0, 50.0 * temp, tol, max_depth, temp, params);
    }
    // -------------------------------------



    double EquilibriumEnergyDensityAux(double p, double temp, SP& params)
    {
        double m = params.mass;
        double Ep = sqrt(p * p + m * m);
        double f_equilibrium = exp(-Ep / temp);

        return 2.0 * p * p * Ep * f_equilibrium / (4.0 * PI * PI );
    }
    // -------------------------------------



    double GetTemperature(double z, SP& params)
    {
        double tau_0 = params.tau_0;
        double step_size = params.step_size;

        int n = (int) floor((z - tau_0) / step_size);	
        double z_frac = (z - tau_0) / step_size - (double) n;

        return 1.0 / ((1.0 - z_frac) * params.D[n] + z_frac * params.D[n+1]);
    }
    // -------------------------------------



    double InvertEnergyDensity(double e, SP& params)
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
            double e1 = EquilibriumEnergyDensity(x1, params);


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



    double TauRelaxation(double tau, SP& params)
    {
        double T = GetTemperature(tau, params);
        double eta_s = params.eta_s;
        return 5.0 * eta_s / T;
    }
    // -------------------------------------



    double H2(double alpha, double zeta)
    {
        if (abs(alpha - 1.0) < eps) alpha = 1.0 - eps;

        double beta;

        if (abs(alpha) < 1)
        {
            beta = sqrt((1.0 - alpha * alpha) / (alpha * alpha + zeta * zeta));
            return alpha * (sqrt(alpha * alpha + zeta * zeta) +  (1.0 + zeta * zeta) / sqrt(1.0-alpha * alpha) * atan(beta));
        }
        else if (abs(alpha) > 1)
        {
            beta = sqrt( (alpha*alpha - 1)/(alpha*alpha + zeta*zeta) );
            return ( alpha*(sqrt(alpha*alpha + zeta*zeta) +  (1.0 + zeta*zeta)/sqrt(alpha*alpha-1)*atanh(beta) )  );
        }
        else return 0;
    }
    // -------------------------------------



    double H2Tilde(double y, double z)
    {
        return GausQuad(H2TildeAux, 0, 50.0, tol, max_depth, y, z);
    }
    // -------------------------------------
    


    double H2TildeAux(double u, double y, double z)
    {
        if (abs(u) < eps) u = eps;
        return u * u * u * exp(-sqrt(u * u + z * z)) * H2(y, z / u);
    }
    // -------------------------------------

    

    double EquilibriumDistribution(double tau, double z, SP& params)
    {
        double m = params.mass;
        double T = GetTemperature(z, params);
        double gamma = z / tau;
        double zeta = m / T;

        double integrand_z = pow(T, 4.0) / (4.0 * PI * PI) * H2Tilde(gamma,zeta);


        return integrand_z;
    }
    // -------------------------------------



    double InitialDistribution(double theta, double p, double tau, SP& params)
    {
        double m        = params.mass;
        double tau_0    = params.tau_0;
        double xi_0     = params.xi_0;
        double alpha_0  = params.alpha_0;
        double Lambda_0 = params.Lambda_0;

        double Ep = sqrt(p * p + m * m);
        double k = tau / tau_0;
        double Ep3 = sqrt(p * p * sin(theta) * sin(theta) + (1.0 + xi_0) * p * p * cos(theta) * cos(theta) * k * k + m * m);
        double f_in3 = (1.0 / alpha_0) * exp(-Ep3 / Lambda_0);
        
        return 2.0 * p * p * Ep * sin(theta) * f_in3 / (4.0 * PI * PI);
    }
    // -------------------------------------



    double DecayFactor(double tau1, double tau2, SP& params)
    {
        double val = GausQuad([](double x, SP& params){ return 1.0 / TauRelaxation(x, params); }, tau2, tau1, tol, max_depth2, params);
        return exp(-val);
    }
    // -------------------------------------



    double EquilibriumContribution(double tau, SP& params)
    {
        double tau_0 = params.tau_0;
        return GausQuad(EquilibriumContributionAux, tau_0, tau, tol, max_depth2, tau, params);
    }
    // -------------------------------------



    double EquilibriumContributionAux(double x, double tau, SP& params)
    {
        return DecayFactor(tau, x, params) * EquilibriumDistribution(tau, x, params) / TauRelaxation(x, params);
    }
    // -------------------------------------



    double InitialEnergyDensity(double tau, SP& params)
    {
        double tau_0    = params.tau_0;
        double xi_0     = params.xi_0;
        double m        = params.mass;
        double alpha_0  = params.alpha_0;
        double Lambda_0 = params.Lambda_0;

        // analytic result
        double y0 = tau_0 / (tau * sqrt(1.0 + xi_0));
        double z0 = m / Lambda_0;

        return pow(Lambda_0, 4.0) * H2Tilde(y0, z0) / (4.0 * PI * PI * alpha_0);
    }
    // -------------------------------------



    double GetMoments(double tau, SP& params)
    {
        double tau_0 = params.tau_0;
        return DecayFactor(tau, tau_0, params) * InitialEnergyDensity(tau, params) + EquilibriumContribution(tau, params);
    }
    // -------------------------------------



    void Run(std::ostream& out, SP& params)
    {
        double tau_0     = params.tau_0;
        int steps        = params.steps;
        double step_size = params.step_size;
        
        // Setting upp initial guess for temperature
        double e0 = InitialEnergyDensity(tau_0, params);
        double T0 = InvertEnergyDensity(e0, params);

        for (int i = 0; i < steps; i++)
        {
            double tau = tau_0 + (double)i * step_size;
            params.D[i] = 1.0 / T0 / pow(tau_0 / tau, 1.0 / 3.0);
        }


        // Iteratively approximate temperature evolution
        int n = 0;
        double err = inf;
        double last_D = 1.0 / T0;
        std::vector<double> e(steps);
        do
        {
            std::cout << "n: " << n << std::endl;
            double tau = tau_0;

            // omp_set_dynamic(0);   // Ensure no subthreading
            // omp_set_num_threads(6);
            // int i;
            // #pragma omp parallel for shared(e, steps) private(i) schedule(static,1)
            for (int i = 0; i < steps; i++)
            {
                e[i] = GetMoments(tau, params);
                double Temp = InvertEnergyDensity(e[i], params);
                params.D[i] = 1.0 / Temp;
                tau += step_size;
            }
            // Note: including the params.D update statement int the loop above, leads to a much faster
            // convergence. Is this a bug, or a feature?

            // for (int i = 0; i < steps; i++)
            // {
            //     double Temp = InvertEnergyDensity(e[i], params);
            //     Print(out, tau, Temp);
            //     params.D[i] = 1.0 / Temp;
            // }
            err = abs(last_D - params.D[steps - 1]) / params.D[steps - 1];
            last_D = params.D[steps - 1];
            n++;
        } while (err > eps);   
        
        for (int i = 0; i < steps; i++)
        {
            double tau = tau_0 + (double)i * step_size;
            Print(out, std::setprecision(16), tau, params.D[i], 1.0/params.D[i]*.197); 
        }
    }
}

