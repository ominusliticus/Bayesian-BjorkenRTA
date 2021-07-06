//
// Author: Kevin Ingles

#include "Errors.hpp"
#include "Integration.hpp"

#include <vector>
#include <fstream>

namespace bayes
{
    enum class Prior { UNI = 0, GAUS, SYM, CONJ };
    class BayesianParameterEstimation
    {
    public:
        BayesianParameterEstimation() = default;
        BayesianParameterEstimation(const char* output_file, std::vector<double> x_data, std::vector<double> y_data, std::vector<double> y_data_err);
        
        double LogLikelihood(double x_data, double y_data, double y_data_err, std::vector<double> param_values);
        double LogPrior(std::vector<double> param_limits, Prior prior);
        double LogPosterior(std::vector<double> param_values, double x_data, double y_data, double y_data_err, Prior prior);

        // RunMCMC samples the posterior and then outputs the data to a file in the format
        // x_data_point log_posterior_eval sampled_paramater_values. . .
        void RunMCMC(int burnin_steps, int steps, int param_count);

        // For convenience, this outputs python code that will make the corner plots 
        void GeneratPythonCode(int param_count);
    private:
        std::fstream _fout;
        std::vector<double> _x_data;
        std::vector<double> _y_data;
        std::vector<double> _y_data_err;
    };
}