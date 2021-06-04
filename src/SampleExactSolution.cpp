//
// Author: Kevin Ingles

#include "../include/SampleExactSoln.hpp"


Particle ProcessSample(SP& params, double tau, double eta, double pT, double w)
{
    return Particle(params.mass, tau, eta, pT, w);
}