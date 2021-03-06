//
// Author: Kevin Ingles

#ifndef HYDRO_THEORIES_HPP
#define HYDRO_THEORIES_HPP

#include "Errors.hpp"
#include "Parameters.hpp"
#include "GlobalConstants.hpp"
#include "Integration.hpp"

#include <vector>

using SP = SimulationParameters;

namespace hydro
{
    enum class theory {CE = 0, DNMR};
    struct ViscousHydroEvolution
    {
        // Constructors
        ViscousHydroEvolution() = default;

        // Setup and run numerical evolution
        void RunHydroSimulation(SP& params, theory theo);

        // Need to invert enery density to get temperature. This is done by taking advantage of
        // the Landau matching condition, i.e. the energy denisty in the comoving frame is 
        // the same as the equilibrium energy density
        double EquilibriumEnergyDensity(double temp, SP& params);
        double InvertEnergyDensity(double e, SP& params);

        struct TransportCoefficients
        {
            double tau_pi;
            double beta_pi;
            double tau_Pi;
            double beta_Pi;
            double delta_pipi;
            double delta_PiPi;
            double lambda_piPi;
            double lambda_Pipi;
            double tau_pipi;
        };
        TransportCoefficients CalculateTransportCoefficients(double e, double pi, double Pi, SP& params, theory theo);

        // Evolution equations
        double  dedt(double e, double p,  double pi, double Pi,  double tau);
        double dpidt(double pi, double Pi, double tau, TransportCoefficients& tc);
        double dPidt(double pi, double Pi, double tau, TransportCoefficients& tc);
        
        // Dynamical variables for RK4
        double   e1,   e2,   e3,   e4;
        double   p1,   p2,   p3,   p4;
        double  pi1,  pi2,  pi3,  pi4;
        double  Pi1,  Pi2,  Pi3,  Pi4;
        double  de1,  de2,  de3,  de4;
        double dpi1, dpi2, dpi3, dpi4;
        double dPi1, dPi2, dPi3, dPi4;
    };



    struct AnisoHydroEvolution
    {
        // Constructors
        AnisoHydroEvolution() = default;

        // Setup and run numerical evolution
        void RunHydroSimulation(SP& params);

        // Need to invert enery density to get temperature. This is done by taking advantage of
        // the Landau matching condition, i.e. the energy denisty in the comoving frame is 
        // the same as the equilibrium energy density
        double EquilibriumEnergyDensity(double temp, SP& params);
        double InvertEnergyDensity(double e, SP& params);

        struct TransportCoefficients
        {
            double tau_pi;
            double tau_Pi;
            double zetaBar_zT;
            double zetaBar_zL;
        };
        TransportCoefficients CalculateTransportCoefficients(double e, double p, double pt, double pl, SP& params);
        // Functions used to calcualte the transport coefficients
        double InvertShearToXi(double e, double p, double pi);
        double R200(double xi);
        double R201(double xi);
        double R220(double xi);
        double R221(double xi);
        double R240(double xi);

        // Evolution equations
        double  dedt(double e, double pl, double tau);
        double dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc);
        double dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc);

        // Dynamic variables for RK4: allocate here to make sure CPU has to constantly allocate new memory
        double   e1,   e2,   e3,   e4;
        double   p1,   p2,   p3,   p4;
        double  pt1,  pt2,  pt3,  pt4;
        double  pl1,  pl2,  pl3,  pl4;
        double  de1,  de2,  de3,  de4;
        double dpt1, dpt2, dpt3, dpt4;
        double dpl1, dpl2, dpl3, dpl4;

        // Simulation information
        double T0;      // Starting temperature in fm^{-1}
    };
}

#endif 