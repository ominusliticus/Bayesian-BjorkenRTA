// 
// Author: Kevin Ingles

#ifndef SAMPLE_EXACT_SOLN_HPP
#define AMPLE_EXACT_SOLN_HPP

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"
#include "ExactSolution.hpp"
#include "Particle.hpp"
#include "Momentum.hpp"

#include <vector>
#include <random>

class Sampler
{
    public:
    using SP = SimulationParameters;

    // This function abstract the call to instantiating a Particle object
    // Its primary purpose is to create an instance of Particle in the Metropolis sampling routine
    Particle ProcessSample(SP& params, double tau, double eta, double w, double pT);

    // Sample LRF momentum using Scott-Pratt trick
    LRFMomentum SampleLRFMomentum(std::default_random_engine generator, double mass, double Temperature);

    // Exact solutoin has contributions from two separate terms
    // We must sample both separately, as they are significanlty different distributions
    void SampleFisrtTerm(std::vector<MilneMomentum> sample_momenta);
    void SampleSecondTerm(std::vector<MilneMomentum> sample_momenta);
    void SampleExactSolution(void);

    private:
    std::vector<double> _tau_arr;               // Vector containing the tau points in evolution
    std::vector<double> _temperature_arr;       // Vecotr containing the corresponding temperature (see comment above)
    std::vector<Particle> _sampled_particles;   // Vector containing sampled particles 

    int samples;        // number of samples taken: used to calculate sampler efficiency
    int acceptances;    // number of samples kept: used to calculate sample efficiency

    SimulationParameters _params;

    double _tau_f;                  // Sampling time (more generically freeze out time)
    double _temperature_f;          // Sampling temperature (more genericaly freeze out temperature)
    double _total_energy_density;   // Total energy density at time tau_f

    // Helper functions that don't need to be public
    double GaussianKernel(MilneMomentum p_mu);
};

#endif