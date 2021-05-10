#include "../include/ExactSolution.hpp"
#include "../include/Integration.hpp"

#include <iomanip>

// Note: that the integration functions are written like this to make sure the 
// integration routine GausQuad is aware of th calss instance and can see that functions

namespace exact{

    const double tol = 1e-8;
    const int max_depth = 1;

    
    ExactSolution::~ExactSolution()
    {
        _D.clear();
    }
    //----------------------------------------



    double ExactSolution::EquilibriumEnergyDensity(double T_eq)
    {
        return GausQuad(
            [this](double p, double T, double m) {return EquilibriumEnergyDensityAux1(p, T, m); },
            0, 50 * T_eq, tol, max_depth, T_eq, _mass);
    }
    //----------------------------------------



    double ExactSolution::EquilibriumEnergyDensityAux1(double p, double T_eq, double mass)
    {
        return GausQuad(
            [this](double th, double p, double T, double m){ return EquilibriumEnergyDensityAux2(th, p, T, m); }, 
            0, PI/2, tol, max_depth, p, T_eq, mass);
    }
    //----------------------------------------



    double ExactSolution::EquilibriumEnergyDensityAux2(double theta, double p, double T_eq, double mass) 
    {
        double Eq = sqrt(p * p + mass * mass);
        double f_eq = exp(-Eq / T_eq);
        return 2 * p * p * sin(theta) * f_eq / (4.0 * PI * PI);
    }
    //----------------------------------------



    double ExactSolution::GetTemperature(double z)
    {
        int n = (int) floor((z - _tau_0) / _step_size);
        double z_frac = (z - _tau_0) / _step_size - (double) n;
        return 1.0 / ((1.0 - z_frac) * _D[n] + z_frac * _D[n+1]);
    }
    //----------------------------------------



    double ExactSolution::TauRelaxation(double tau)
    {
        double T = GetTemperature(tau);
        return 5.0 * _eta_s / T;
    }
    //----------------------------------------



    double ExactSolution::H(double alpha)
    {
        if (abs(alpha - 1) < eps)
            alpha = 1.0 - eps;

        double beta = sqrt((1.0 - alpha * alpha) / (alpha * alpha));
        return alpha * (alpha + atan(beta) / sqrt(1 - alpha * alpha));
    }
    //----------------------------------------

    double ExactSolution::H2(double alpha, double zeta)
    {
        if (abs(alpha - 1.0)<eps)
            alpha = 1.0 - eps;

        double beta;

        if (abs(alpha) < 1)
        {
            beta = sqrt((1.0 - alpha * alpha) / (alpha * alpha + zeta * zeta));
            return (alpha * (sqrt(alpha * alpha + zeta * zeta) +  (1.0 + zeta * zeta) / sqrt(1.0 - alpha * alpha) * atan(beta)));
        }

        else if (abs(alpha)>1)
        {
            beta = sqrt((alpha*alpha - 1)/(alpha*alpha + zeta*zeta) );
            return ( alpha*(sqrt(alpha*alpha + zeta*zeta) +  (1.0 + zeta*zeta)/sqrt(alpha*alpha-1)*atanh(beta) )  );
        }
    }


    double ExactSolution::EquilibriumDistribution(double tau, double z)
    {
        double gamma = z / tau;
        double T = GetTemperature(z);
        double integrand_z = 6.0 * pow(T, 4.0) * H(gamma) / (4.0 * PI * PI);
        return integrand_z;
    }
    //----------------------------------------



    double ExactSolution::InitialDistribution(double theta, double p, double tau, double mass)
    {
        double Ep = sqrt(p * p + mass * mass);
        double k = tau / _tau_0;

        double Ep3 = sqrt(p * p * sin(theta) * sin(theta) + (1 + _xi_0) * p * p * cos(theta) * cos(theta) * k * k);
        double f_0 = exp(-Ep3 / _Lambda_0);
        return 2.0 * p * p * Ep * sin(theta) * f_0 / (4.0 * PI * PI);
    }
    //----------------------------------------



    double ExactSolution::DecayFactor(double tau1, double tau2)
    {
        double val = GausQuad([this](double x){ return 1.0 / TauRelaxation(x); }, tau2, tau1, tol, 5);
        return exp(-val);
    }
    //----------------------------------------



    double ExactSolution::IntegralOfEquilibriumContribution(double tau)
    {
        return GausQuad(
            [this](double x, double t, double m){ return IntegralOfEquilibriumContributionAux(x, t, m); }, 
            _tau_0, tau, tol, 5, tau, _mass);
    }
    //----------------------------------------



    double ExactSolution::IntegralOfEquilibriumContributionAux(double x, double tau, double mass)
    {
        return 1.0 / TauRelaxation(x) * DecayFactor(tau, x) * EquilibriumDistribution(tau, x);
    }
    //----------------------------------------



    double ExactSolution::InitialEnergyDensity(double tau)
    {
        return GausQuad(
            [this](double p, double t, double m){ return InitialEnergyDensityAux(p, t, m); }, 
            0, 50 * _Lambda_0, tol, max_depth, tau, _mass);
    }
    //----------------------------------------



    double ExactSolution::InitialEnergyDensityAux(double p, double tau, double mass)
    {
        return GausQuad(
            [this](double th, double p, double t, double m){ return InitialDistribution(th, p, t, m); }, 
            0, PI / 2, tol, max_depth, p, tau, mass);
    }
    //----------------------------------------
    
    

    double ExactSolution::GetMoments(double tau)
    {
        double moment = DecayFactor(tau, _tau_0) * 6.0 * pow(_Lambda_0, 4.0) * H(_tau_0 / tau /sqrt(1 + _xi_0)) / ( 4.0 * PI * PI);
        moment += IntegralOfEquilibriumContribution(tau);
        return moment;
    }
    //----------------------------------------



    void ExactSolution::Run(int iters, std::ostream& out)
    {
        // Setting upp initial guess for temperature
        double e0 = InitialEnergyDensity(_tau_0);
        double T0 = pow(PI * PI * e0 / 3.0, 0.25);

        for (int i = 0; i < _steps; i++)
        {
            double tau = _tau_0 + (double)i * _step_size;
            _D[i] = 1.0 / T0 / pow(_tau_0 / tau, 1.0 / 3.0);
        }


        // Iteratively approximate temperature evolution
        int n = 0;
        double tau = 0.0;
        std::vector<double> e(_steps);
        do
        {
            std::cout << "n: " << n << std::endl;

            tau = _tau_0;
            for (int i = 0; i < _steps; i++)
            {
                e[i] = GetMoments(tau);
                double Temp = pow(PI * PI * e[i] / 3.0, 0.25);
                _D[i] = 1.0 / Temp;
                tau += _step_size;
            }
            n++;
        } while (n <= iters);   
        
        for (int i = 0; i < _steps; i++)
        {
            tau = _tau_0 + (double)i * _step_size;
            out << std::setprecision(16) << tau << " " << _D[i] << " " << 1.0/_D[i]*.197 << std::endl; 
        }
    }
}
