// 
// Author: Kevin Ingles

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"
#include "ExactSolution.hpp"
#include "Particle.hpp"

#include <vector>

struct Sampler
{
    Sampler() = defualt;
    Sampler(long points);
    ~Sampler();

    std::vector<Particle> particles;
};