//
// Author: Kevin Ingles

#include "../include/Momentum.hpp"

#include <math.h>

MilneMomentum::MilneMomentum(LRFMomentum pLRF, double tau)
{
    double tau2 = tau * tau;
    double pxLRF = pLRF.px;
    double pyLRF = pLRF.py;
    double pzLRF = pLRF.pz;
    double ELRF  = pLRF.E;

    double pxLRF2 = pxLRF * pxLRF;
    double pyLRF2 = pyLRF * pyLRF;
    double pzLRF2 = pzLRF * pzLRF;
    double ELRF2  = ELRF  * ELRF;

    pT = sqrt(pxLRF2 + pyLRF2);
    w  = pzLRF / tau2;

    double m = sqrt(ELRF2 - pxLRF2 - pyLRF2 - pzLRF2);

    v = sqrt(w * w + (pT * pT + m * m) * tau2);

    if (ELRF / tau2 != v)
        Print_Error(std::cerr, "Error converting LRF momentum to Milne momentum.");
}