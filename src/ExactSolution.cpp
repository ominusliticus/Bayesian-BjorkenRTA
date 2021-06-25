// 
// Author: Kevin Ingles
// Credit: Chandrodoy Chattopdhyay for first version
#include "../include/ExactSolution.hpp"
#include "../include/Integration.hpp"

#include <iomanip>
#include <cmath>
#include <fstream>



// Note: that the integration functions are written like this to make sure the 
// integration routine GausQuad is aware of th calss instance and can see that functions

namespace exact{
    
    // const double tol = eps;
    // const int max_depth = 5;
    // const int max_depth2 = 5;
    const double tol     = eps; //eps;
    const int max_depth  = 0;
    const int max_depth2 = 1;




    ////////////////////////////////////////////////////////////
    ///        Code need by all subsequent functions         ///
    ////////////////////////////////////////////////////////////
    
    double GetTemperature(double z, SP& params)
    {
        double tau_0 = params.tau_0;
        double step_size = params.step_size;

        int n = (int) floor((z - tau_0) / step_size);	
        double z_frac = (z - tau_0) / step_size - (double) n;

        return 1.0 / ((1.0 - z_frac) * params.D[n] + z_frac * params.D[n+1]);
    }
    // -------------------------------------


    
    double TauRelaxation(double tau, SP& params)
    {
        double T = GetTemperature(tau, params);
        double eta_s = params.eta_s;
        return 5.0 * eta_s / T;
    }
    // -------------------------------------



    double DecayFactor(double tau1, double tau2, SP& params)
    {
        double val = GausQuad([](double x, SP& params){ return 1.0 / TauRelaxation(x, params); }, tau2, tau1, tol, max_depth2, params);
        return std::exp(-val);
    }
    // -------------------------------------




    ////////////////////////////////////////////////////////////
    ///        Code for evaluating moments analytically      ///
    ////////////////////////////////////////////////////////////

    double H(double y, double z, Moment flag)
    {
        if (std::abs(y - 1.0) < eps) y = 1.0 - eps;

        double x;
        double atanh_val;
        double sign;

        // Checks to ensure H returns a real value
        if (std::abs(y) < 1)
        {
            x = 1.0 - y * y;
            atanh_val = std::atan(std::sqrt((1.0 - y * y) / (y * y + z * z)));
            sign = -1.0; // from factoring out factors of i
        }
        else if (std::abs(y) > 1)
        {
            x = y * y - 1.0;
            atanh_val = std::atanh(std::sqrt((y * y - 1) / (y * y + z * z)));
            sign = 1.0;
        }
        else return 0;
        
    
        switch (flag) // Note Moment::PEQ gets switched to ED in EquilibriumDistributionMoment
        {
            case Moment::ED:
                return y * (std::sqrt(y * y + z * z) + (1 + z * z) / std::sqrt(x) * atanh_val);
            
            case Moment::PL:
                return sign * y * y * y / std::pow(x, 1.5) * (std::sqrt(x *(y * y + z * z)) - (z * z + 1.0) * atanh_val);

            case Moment::PT:
                return sign * y / std::pow(x, 1.5) * (-std::sqrt(x * (y * y + z * z)) + ( z * z + 2.0 * y * y - 1.0) * atanh_val);
        }

        return -888;
    }
    // -------------------------------------



    double HTilde(double y, double z, Moment flag)
    {
        return GausQuad(HTildeAux, 0, inf, tol, max_depth, y, z, flag);
    }
    // -------------------------------------
    


    double HTildeAux(double u, double y, double z, Moment flag)
    {
        if (std::abs(u) < eps) u = eps;
        return u * u * u * std::exp(-std::sqrt(u * u + z * z)) * H(y, z / u, flag);
    }
    // -------------------------------------

    

    double EquilibriumDistributionMoment(double tau, double zeta, SP& params, Moment flag)
    {
        double m = params.mass;
        double T = GetTemperature(zeta, params);
        double y = zeta / tau;
        double z = m / T;

        switch (flag)
        {
            case Moment::ED:
                return std::pow(T, 4.0) / (4.0 * PI * PI) * HTilde(y, z, Moment::ED);
            
            case Moment::PL:
                return std::pow(T, 4.0) / (4.0 * PI * PI) * HTilde(y, z, Moment::PL);

            case Moment::PT:
                return std::pow(T, 4.0) / (8.0 * PI * PI) * HTilde(y, z, Moment::PT);

            case Moment::PEQ:
                return 0.0;
        }

        return -1111;
    }
    // -------------------------------------



    double InitialDistributionMoment(double tau, SP& params, Moment flag)
    {
        double tau_0    = params.tau_0;
        double xi_0     = params.xi_0;
        double m        = params.mass;
        double alpha_0  = params.alpha_0;
        double Lambda_0 = params.Lambda_0;

        // analytic result
        double y0 = tau_0 / (tau * std::sqrt(1.0 + xi_0));
        double z0 = m / Lambda_0;

        switch (flag)
        {
            case Moment::ED:
                return std::pow(Lambda_0, 4.0) / (4.0 * PI * PI * alpha_0) * HTilde(y0, z0, Moment::ED);
            
            case Moment::PL:
                return std::pow(Lambda_0, 4.0) / (4.0 * PI * PI * alpha_0) * HTilde(y0, z0, Moment::PL);

            case Moment::PT:
                return std::pow(Lambda_0, 4.0) / (8.0 * PI * PI * alpha_0) * HTilde(y0, z0, Moment::PT);

            case Moment::PEQ:
                // PEQ requires us to take moment w.r.t. equilibrium distribution
                double T = GetTemperature(tau, params);
                double z = params.mass / T;
                // Notice that we divide by DecayFactor(...) to offset multiplying by it in GetMomets(...)
                return z * z * std::pow(T, 4.0) / (2.0 * PI * PI) * std::cyl_bessel_k(2, z) / DecayFactor(tau, params.tau_0, params);
        }

        return -2222;
    }
    // -------------------------------------
    
    
    
    double EquilibriumContribution(double tau, SP& params, Moment flag)
    {
        return GausQuad(EquilibriumContributionAux, params.tau_0, tau, tol, max_depth2, tau, params, flag);
    }
    // -------------------------------------



    double EquilibriumContributionAux(double x, double tau, SP& params, Moment flag)
    {
        return DecayFactor(tau, x, params) * EquilibriumDistributionMoment(tau, x, params, flag) / TauRelaxation(x, params);
    }
    // -------------------------------------



    double GetMoments(double tau, SP& params, Moment flag)
    {
        return DecayFactor(tau, params.tau_0, params) * InitialDistributionMoment(tau, params, flag) + EquilibriumContribution(tau, params, flag);
    }
    // -------------------------------------




    ////////////////////////////////////////////////////////////
    ///        Code for evaluating moments numerically       ///
    ////////////////////////////////////////////////////////////

    double InitialDistribution(double w, double pT, SP& params)
    {
        double m        = params.mass;
        double tau_0    = params.tau_0;
        double xi_0     = params.xi_0;
        double alpha_0  = params.alpha_0;
        double Lambda_0 = params.Lambda_0;

        double vp = std::sqrt((1 + xi_0) * w * w  + (pT * pT + m * m) * tau_0 * tau_0);
        double f_in3 = (1.0 / alpha_0) * std::exp(-vp / (Lambda_0 * tau_0));
        
        return f_in3 / (4.0 * PI * PI);
    }
    // -------------------------------------



    double EquilibriumDistribution(double w, double pT, double tau, SP& params)
    {
        double T   = GetTemperature(tau, params);
        double m   = params.mass;
        double vp  = std::sqrt(w * w  + (pT * pT + m * m) * tau * tau);
        double feq = std::exp(-vp / (T * tau));

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

            case Moment::PEQ:
                Print_Error(std::cerr, "Error in exact::GetMoment2: Moment::PEQ option not available.");
                break;
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
                    double vp = std::sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return 2.0 * vp * EaxctDistribution(w, pT, tau, params) / (tau * tau) ; 
                }, 0, inf, tol, max_depth2, pT, tau, params);

            case Moment::PL:
                return GausQuad([](double w, double pT, double tau, SP&params){ 
                    double m = params.mass;
                    double vp = std::sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return 2.0 * w * w * EaxctDistribution(w, pT, tau, params) / (tau * tau * vp) ; 
                }, 0, inf, tol, max_depth2, pT, tau, params);
                
            case Moment::PT:
                return GausQuad([](double w, double pT, double tau, SP&params){ 
                    double m = params.mass;
                    double vp = std::sqrt(w * w + (pT * pT + m * m) * tau * tau);
                    return EaxctDistribution(w, pT, tau, params) / vp; 
                }, 0, inf, tol, max_depth2, pT, tau, params);

            case Moment::PEQ:
                Print_Error(std::cerr, "Error in exact::GetMoment2Aux: Moment::PEQ option not available.");
                break;
        }
        return -999;
    }
    // -------------------------------------



    ////////////////////////////////////////////////////////////
    ///        Code to solve temperature evolution           ///
    ////////////////////////////////////////////////////////////

    double EquilibriumEnergyDensity(double temp, SP& params)
    {
        double z = params.mass / temp;
        return 3.0 * std::pow(temp, 4.0) / (PI * PI) * (z * z * std::cyl_bessel_k(2, z) / 2.0 + z * z * z * std::cyl_bessel_k(1, z) / 6.0);
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
            double e1    = EquilibriumEnergyDensity(x1, params);


            if (std::abs(e_mid - e) < prec) 
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
                if (std::abs(copy - mid) < prec)
                flag_1 = 1;	
                copy = mid;
            }
        }while (flag_1 != 1 && n <= 2000);

        return mid;	
    }
    // -------------------------------------



    void Run(const char* file_name, SP& params)
    {
        double tau_0     = params.tau_0;
        int steps        = params.steps;
        double step_size = params.step_size;
        
        // Setting upp initial guess for temperature
        double e0 = InitialDistributionMoment(tau_0, params, Moment::ED);
        double T0 = InvertEnergyDensity(e0, params);

        for (int i = 0; i < steps; i++)
        {
            double tau = tau_0 + (double)i * step_size;
            params.D[i] = 1.0 / T0 / std::pow(tau_0 / tau, 1.0 / 3.0);
        }


        // Iteratively solve for temperature evolution
        int n = 0;
        double err = inf;
        double last_D = 1.0 / T0;
        std::vector<double> e(steps);
        Print(std::cout, "Calculating temperature evoluton.");
        do
        {
            Print(std::cout, fmt::format("n = {}",n));
            double tau = tau_0;

            for (int i = 0; i < steps; i++)
            {
                e[i] = GetMoments(tau, params, Moment::ED);
                tau += step_size;
            }

            for (int i = 0; i < steps; i++)
            {
                double Temp = InvertEnergyDensity(e[i], params);
                params.D[i] = 1.0 / Temp;
            }
            err = std::abs(last_D - params.D[steps - 1]) / params.D[steps - 1];
            last_D = params.D[steps - 1];
            n++;
        } while (err > eps);   
        
        std::fstream out(file_name, std::ios::out);
        for (int i = 0; i < steps; i++)
        {
            double tau = tau_0 + (double)i * step_size;
            Print(out, std::setprecision(16), tau, params.D[i], 1.0/params.D[i]*.197); 
        }
        out.close();
        Print(std::cout, "Temperature evolution calculation terminated successfully.");
    }
    //--------------------------------------



    void OutputMoments(const char* file_name, SP& params)
    {
        Print(std::cout, "Calculating moments of distribution function.");
        std::fstream fout(file_name, std::fstream::out);
        fout << std::fixed << std::setprecision(16);
        for (double tau = params.ll; tau <= params.ul + params.step_size; tau += params.step_size)
        {
            double new_e_density = exact::GetMoments(tau, params, exact::Moment::ED);
            double new_pL        = exact::GetMoments(tau, params, exact::Moment::PL);
            double new_pT        = exact::GetMoments(tau, params, exact::Moment::PT);
            double new_peq       = exact::GetMoments(tau, params, exact::Moment::PEQ);
            Print(fout, tau, new_e_density, new_pL, new_pT, new_peq);
        }
        fout.close();
    }
}
