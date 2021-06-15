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
        
        // Dynamical variables for RK4
        double   e1,   e2,   e3,   e4;
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

        // Anisotropic parameters (or variables as I have called them below)
        struct AnisoVariables
        {
            double Lambda;
            double alpha_T;
            double alpha_L;
        };

        // Functions for inverting the energy density, transverse pressure and longitudinal pressure
        // to get the parameters Lambda, alpha_L and alpha_T at each time step
        // These parameters are necessary to calculate the transport coeffiecents
        // Code adopted from https://github.com/mjmcnelis/cpu_vah.git
        void ComputeF(double e, double pt, double pl, double mass, double (&X)[3], double F[3]);    // The weird syntax here, (&X)[3] is to allow us to use structured bindings
                                                                                                    // This is a patch I had to introduce due to poor initial planning.
        void ComputeJ(double e, double pt, double pl, double mass, double (&X)[3], double F[3], double (*J)[3]);
        double LineBackTrack(double e, double pt, double pl, double mass, const double Xcurrent[3], const double dX[3], double dXmagnitude, double g0, double F[3]);
        void FindAnisoVariables(double e, double pt, double pl, double mass, AnisoVariables aVars); // Algorithm akin to a gradient decent, used to do Netwon-Coats methods
                                                                                                    // in more than 1 dimension

        // Anisotropic Integrals needed for calculating transport coefficients,
        // and also for the inversion (e, PT, PL) --> (Lambda, alpha_T, alpha_L)
        double IntegralI(int n, int r, int q, int s, double mass, double (&X)[3]);
        double IntegralIAux(int n, int r, int q, int s, double mass, double pbar, double (&X)[3]);
        double IntegralJ(int n, int r, int q, int s, double mass, double (&X)[3]);
        double IntegralJAux(int n, int r, int q, int s, double mass, double pbar, double (&X)[3]);
        double IntegrandR(int n, int r, int q, double mass, double pbar, double (&X)[3]);

        // Need to invert enery density to get temperature, this is done by taking advantage of
        // the Landau matching condition, i.e. the energy denisty in the comoving frame is 
        // the same as the equilibrium energy density
        // (Adopted from ExactSolution.hpp)
        double EquilibriumEnergyDensity(double temp, SP& params);
        double InvertEnergyDensity(double e, SP& params);

        struct TransportCoefficients
        {
            double tau_pi;
            double tau_Pi;
            double zetaBar_zL;
            double zetaBar_zT;
        };
        TransportCoefficients& CalculateTransportCoefficients(double e, double pt, double pl, SP& params, AnisoVariables aVars);
        // Evolution equations
        double dedt(double e, double pl, double tau);
        double dpldt(double p, double pt, double pl, double tau, TransportCoefficients& tc);
        double dptdt(double p, double pt, double pl, double tau, TransportCoefficients& tc);

        // Dynamic variables for RK4: allocate here to make sure CPU has to constantly allocate new memory
        double   e1,   e2,   e3,   e4;
        double  pt1,  pt2,  pt3,  pt4;
        double  pl1,  pl2,  pl3,  pl4;
        double  de1,  de2,  de3,  de4;
        double dpt1, dpt2, dpt3, dpt4;
        double dpl1, dpl2, dpl3, dpl4;
        AnisoVariables aVars;

        // Simulation information
        double T0;      // Starting temperature in fm^{-1}
    };
}

#endif 