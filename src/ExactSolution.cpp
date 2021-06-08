// 
// Author: Kevin Ingles
// Credit: Chandrodoy Chattopdhyay for first version
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
        // 
        return GausQuad(EquilibriumEnergyDensityAux, 0.0, inf, tol, max_depth, temp, params);
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
            return alpha * (sqrt(alpha * alpha + zeta * zeta) + (1.0 + zeta * zeta) / sqrt(1.0 - alpha * alpha) * atan(beta));
        }
        else if (abs(alpha) > 1)
        {
            beta = sqrt( (alpha*alpha - 1)/(alpha*alpha + zeta*zeta) );
            return alpha * (sqrt(alpha * alpha + zeta * zeta) + (1.0 + zeta * zeta) / sqrt(alpha * alpha - 1) * atanh(beta));
        }
        else return 0;
    }
    // -------------------------------------



    double H2Tilde(double y, double z)
    {
        return GausQuad(H2TildeAux, 0, inf, tol, max_depth, y, z);
    }
    // -------------------------------------
    


    double H2TildeAux(double u, double y, double z)
    {
        if (abs(u) < eps) u = eps;
        return u * u * u * exp(-sqrt(u * u + z * z)) * H2(y, z / u);
    }
    // -------------------------------------

    

    double EquilibriumEnergyDensity(double tau, double z, SP& params)
    {
        double m = params.mass;
        double T = GetTemperature(z, params);
        double gamma = z / tau;
        double zeta = m / T;

        double integrand_z = pow(T, 4.0) / (4.0 * PI * PI) * H2Tilde(gamma,zeta);


        return integrand_z;
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
        return DecayFactor(tau, x, params) * EquilibriumEnergyDensity(tau, x, params) / TauRelaxation(x, params);
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



    double InitialDistribution(double w, double pT, SP& params)
    {
        double m        = params.mass;
        double tau_0    = params.tau_0;
        double xi_0     = params.xi_0;
        double alpha_0  = params.alpha_0;
        double Lambda_0 = params.Lambda_0;

        double vp = sqrt((1 + xi_0) * w * w  + (pT * pT + m * m) * tau_0 * tau_0);
        double f_in3 = (1.0 / alpha_0) * exp(-vp / (Lambda_0 * tau_0));
        
        return f_in3 / (4.0 * PI * PI);
    }
    // -------------------------------------



    double EquilibriumDistribution(double w, double pT, double tau, SP& params)
    {
        double T   = GetTemperature(tau, params);
        double m   = params.mass;
        double vp  = sqrt(w * w  + (pT * pT + m * m) * tau * tau);
        double feq = exp(-vp / (T * tau));

        return feq / (4.0 * PI * PI);
    }
    // -------------------------------------



    double EaxctDistribution(double w, double pT, double tau, SP& params)
    {
        double tau_0 = params.tau_0;
        double feq_contrib = GausQuad([tau](double t, double pT, double w, SP& params){ 
            return DecayFactor(tau, t, params) * EquilibriumDistribution(w, pT, t, params) / TauRelaxation(t, params);
        }, tau_0, tau, tol, max_depth2, pT, w, params);

        return DecayFactor(tau, tau_0, params) * InitialDistribution(w, pT, params) + feq_contrib;
    }
    // -------------------------------------



    double ThetaIntegratedExactDistribution(double p, double tau, SP& params)
    {
        return GausQuad([](double theta, double p, double tau, SP& params)
        {
            double costheta = cos(theta);
            double sintheta = sin(theta);

            double pz = p * costheta;
            double pT = p * sintheta;
            return sintheta * EaxctDistribution(pz / tau, pT, tau, params);

        }, 0, PI, tol, max_depth, p, tau, params);
    }
    // -------------------------------------



    std::tuple<double, double> EaxctDistributionTuple(double w, double pT, double tau, SP& params)
    {
        double tau_0 = params.tau_0;
        double feq_contrib = GausQuad([tau](double t, double pT, double w, SP& params){ 
            return DecayFactor(tau, t, params) * EquilibriumDistribution(w, pT, t, params) / TauRelaxation(t, params);
        }, tau_0, tau, tol, max_depth2, pT, w, params);
        double initial_contrib = DecayFactor(tau, tau_0, params) * InitialDistribution(w, pT, params);
        return std::make_tuple(initial_contrib, feq_contrib);
    }
    // -------------------------------------



    std::tuple<double, double> ThetaIntegratedExactDistributionTuple(double p, double tau, SP& params)
    {
        double initial_contrib = GausQuad([](double theta, double p, double tau, SP& params)
        {
            double costheta = cos(theta);
            double sintheta = sin(theta);

            double pz = p * costheta;
            double pT = p * sintheta;
            return sintheta * DecayFactor(tau, params.tau_0, params) * InitialDistribution(pz / tau, pT, params);

        }, 0, PI, tol, max_depth, p, tau, params);

        double equilibrium_contrib = GausQuad([](double theta, double p, double tau, SP& params)
        {
            double costheta = cos(theta);
            double sintheta = sin(theta);

            double pz = p * costheta;
            double pT = p * sintheta;
            return sintheta * (GausQuad([tau](double t, double pT, double w, SP& params){ 
                                return DecayFactor(tau, t, params) * EquilibriumDistribution(w, pT, t, params) / TauRelaxation(t, params);
                                }, params.tau_0, tau, tol, max_depth2, pT, pz / tau, params));

        }, 0, PI, tol, max_depth, p, tau, params);

        return std::make_tuple(initial_contrib, equilibrium_contrib);
    }
    // -------------------------------------



    double GetMoments(double tau, SP& params)
    {
        double tau_0 = params.tau_0;
        return DecayFactor(tau, tau_0, params) * InitialEnergyDensity(tau, params) + EquilibriumContribution(tau, params);
    }
    // -------------------------------------



    double GetMoments2(double tau, SP& params, Moment flag)
    {
        switch (flag)
        {
            case Moment::ED:
                return GausQuad([](double pT, double tau, SP& params, Moment flag){
                    return pT * GetMoments2Aux(pT, tau, params, flag); 
                }, 0, inf, tol, max_depth2, tau, params, flag);
            
            case Moment::PL:
                return GausQuad([](double pT, double tau, SP& params, Moment flag)
                {
                    return pT * GetMoments2Aux(pT, tau, params, flag);
                }, 0, inf, tol, max_depth2, tau, params, flag);
            
            case Moment::PT:
                return GausQuad([](double pT, double tau, SP& params, Moment flag)
                {
                    return pT * pT * pT * GetMoments2Aux(pT, tau, params, flag);
                }, 0, inf, tol, max_depth2, tau, params, flag);
        }
        return -999;
    }
    // -------------------------------------



    double GetMoments2Aux(double pT, double tau, SP& params, Moment flag)
    {
        switch (flag)
        {
            case Moment::ED:
                return GausQuad([](double w, double pT, double tau, SP&params){ 
                    double m = params.mass;
                    double vp = sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return 2.0 * vp * EaxctDistribution(w, pT, tau, params) / (tau * tau) ; 
                }, 0, inf, tol, max_depth2, pT, tau, params);

            case Moment::PL:
                return GausQuad([](double w, double pT, double tau, SP&params){ 
                    double m = params.mass;
                    double vp = sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return 2.0 * w * w * EaxctDistribution(w, pT, tau, params) / (tau * tau * vp) ; 
                }, 0, inf, tol, max_depth2, pT, tau, params);
                
            case Moment::PT:
                return GausQuad([](double w, double pT, double tau, SP&params){ 
                    double m = params.mass;
                    double vp = sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return EaxctDistribution(w, pT, tau, params) / vp; 
                }, 0, inf, tol, max_depth2, pT, tau, params);
        }
        return -999;
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


        // Iteratively solve for temperature evolution
        int n = 0;
        double err = inf;
        double last_D = 1.0 / T0;
        std::vector<double> e(steps);
        do
        {
            std::cout << "n: " << n << std::endl;
            double tau = tau_0;

            for (int i = 0; i < steps; i++)
            {
                e[i] = GetMoments(tau, params);
                tau += step_size;
            }

            for (int i = 0; i < steps; i++)
            {
                double Temp = InvertEnergyDensity(e[i], params);
                params.D[i] = 1.0 / Temp;
            }
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
