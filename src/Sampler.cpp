//
// Author: Kevin Ingles

#include "../include/Sampler.hpp"

namespace sampler
{
    std::vector<double> MultivariateNormal::RandomVariates(void)
    {
        std::vector<double> sample(_means.size());
        for (int i = 0; i < _means.size(); i++)
        {
            std::normal_distribution norm(_means[i], _spreads[i]);
            sample[i] = norm(_gen);
        }
        return sample;
    }
}