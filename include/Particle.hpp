// Credit: iS3D/src/cpp/particle.h - https://github.com/derekeverett/iS3D.git

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"

// Struct to contain particle information
// for now we are limited to massive Boltzmann particles or only one species
struct Particle
{
    // Default constructor, so I don't have to manually initialize all data vars
    Particle() = default;
    Particle(double mass, double tau, double eta, double pT, double w)
        : mass{mass}, tau{tau}, eta{eta}, pT{pT} , w{w}
    {
        vp = sqrt(w * w + (pT * pT + mass * mass) * tau * tau);
    };

    double mass;

    // Milne coordinates for particle production point
    double tau;
    double eta;

    // Milne momentum
    double vp;
    double pT;
    double w;
};

#endif 