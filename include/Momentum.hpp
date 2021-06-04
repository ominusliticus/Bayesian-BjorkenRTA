// 
// Author: Kevin Ingles

#ifndef MOMENTUM_HPP
#define MOMENTUM_HPP

#include "Errors.hpp"

// Local Rest Frame mometum in Minkowski coordinates
struct LRFMomentum
{
    LRFMomentum() = default;
    LRFMomentum(double px, double py, double pz, double mass)
        : px{px}, py{py}, pz{pz}
    {
        E = sqrt(px * px + py * py + pz * pz + mass * mass);
    }

    double px;
    double py;
    double pz;
    double E;
};


// Struct that takes in Minkowskiu momenta and converts them to Milne coordinates
struct MilneMomentum
{
    MilneMomentum() = default;
    MilneMomentum(LRFMomentum pLRF, double tau);

    double pT;
    double w;
    double v;
};

#endif