// 
// Author: Kevin Ingles

#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Parameters.hpp"
#include "ExactSolution.hpp"
#include "Particle.hpp"

#include <vector>
#include <random>
#include <type_traits>

namespace sampler
{
    // Class to emulate multi-dimenisonal Gaussian distributions
    class MultivariateNormal
    {
    public:
        MultivariateNormal(std::vector<double> means, std::vector<double> spreads, std::default_random_engine& generator)
            : _means{means}, _spreads{spreads}, _gen{generator} {}
        std::vector<double> RandomVariates(void);

    private: 
        std::vector<double> _means;
        std::vector<double> _spreads;
        std::default_random_engine _gen;
    };


    // I think this is based on the Gibbs samplig method, we can control the spread for each variable
    template <typename Func, class...Args>
    inline std::vector<double> Sampler(Func&& TargetFunc, int num_of_samples, std::vector<double>& start_position, std::vector<double>& proposal_width, Args&&... args)
    {
        // Store starting position in a vector
        std::vector<double> current_vector = start_position;

        // Store jumps sizes in vector
        std::vector<double> spread_vector = proposal_width;

        // Create vector to store samples
        std::vector<std::vector<double>> samples(num_of_samples);
        
        std::default_random_engine generator_proposal(10000);
        std::default_random_engine generator_accept(20000);
        std::normal_distribution standard(0, 1);
        for (int i = 0; i < num_of_samples; i++)
        {
            MultivariateNormal proposal_dist(current_vector, spread_vector, generator_proposal);
            std::vector<double> proposed_vector = proposal_dist.RandomVariates();

            double p_current = TargetFunc(current_vector, std::forward<Args>(args)...);
            double p_next    = TargetFunc(proposed_vector, std::forward<Args>(args)...);

            double p_accept = p_next / p_current;

            if (p_accept < standard(generator_accept))
                current_vector = proposed_vector;

            samples[i] = current_vector;
        }

        return samples;
    }
}

#endif