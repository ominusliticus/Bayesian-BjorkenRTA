//
// Author: Kevin Ingles

#include "config.hpp"
#include "Errors.hpp"
#include "GlobalConstants.hpp"
#include "Integration.hpp"
#include "Parameters.hpp"
#include "ExactSolution.hpp"
#include "HydroTheories.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <functional>
#include <unordered_map>

#if USE_PARALLEL
    #include <omp.h>
#endif

namespace bayes
{
    template<class HydroTheory, class...Args>
    class TestBayesianParameterEstimation
    {
    public:
        TestBayesianParameterEstimation() = default;
        // Need to include the variadic template to allow for passing of hydro::theory enumeration for viscous hydros
        TestBayesianParameterEstimation(SP& params, std::string& file_name, HydroTheory& hydro);
        //TO DO: Add destrucors to clear up memory

        double LogLikelihood(std::vector<double> tau_f, std::vector<double> observable, std::vector<double> observable_err, Args&&... args);
        double LogPrior(void);
        double LogPosterior(double tau_f, double observable, double observable_err);
        void RunMCMC(int burnin, int steps, const char* out_filename, Args&&... args);

    protected:
        SP& _params;
        std::vector<double> _tau_fs;
        std::vector<double> _energy_densities;
        std::vector<double> _energy_density_errors;
        std::unordered_map<std::string, std::vector<double>> _param_limits;

        HydroTheory _model;
        std::vector<double> _log_posterior;
    };

    // enum class Prior { UNI = 0, GAUS, SYM, CONJ };
    // class BayesianParameterEstimation
    // {
    // public:
    //     BayesianParameterEstimation() = default;
    //     BayesianParameterEstimation(const char* output_file, std::vector<double> x_data, std::vector<double> y_data, std::vector<double> y_data_err);
        
    //     virtual double LogLikelihood(double x_data, double y_data, double y_data_err, std::vector<double> param_values) {}
    //     virtual double LogPrior(std::vector<double> param_limits, Prior prior) {}
    //     virtual double LogPosterior(std::vector<double> param_values, double x_data, double y_data, double y_data_err, Prior prior) {}

    //     // RunMCMC samples the posterior and then outputs the data to a file in the format
    //     // x_data_point log_posterior_eval sampled_paramater_values. . .
    //     void RunMCMC(int burnin_steps, int steps, int param_count);

    //     // For convenience, this outputs python code that will make the corner plots 
    //     void GeneratPythonCode(int param_count);
    // protected:
    //     std::fstream _fout;
    //     std::vector<double> _x_data;
    //     std::vector<double> _y_data;
    //     std::vector<double> _y_data_err;
    // };


    // .....oooO0Oooo..........oooO0Oooo..........oooO0Oooo..........oooO0Oooo..........oooO0Oooo.....
    // Begin template implementation
    // .....oooO0Oooo..........oooO0Oooo..........oooO0Oooo..........oooO0Oooo..........oooO0Oooo.....
  
    template<class HydroTheory, class...Args>
    TestBayesianParameterEstimation<HydroTheory, Args...>::TestBayesianParameterEstimation(SP& params, std::string& file_name, HydroTheory& hydro)
        : _params {params}
    {        
        auto file = std::fstream(file_name.data());
        std::string line;
        while (!file.eof())
        {
            std::getline(file, line);
            std::stringstream line_buffer(line);
            double tau;
            double energy_density;
            // double energy_density_error;

            line_buffer >> tau;
            line_buffer >> energy_density;
            // energy_density_error = 1.0;
            // line_buffer >> energy_density_error;

            _tau_fs.push_back(tau);
            _energy_densities.push_back(energy_density);
            _energy_density_errors.push_back(energy_density);
        }
        file.close();

        _model = hydro;

        file = std::fstream("utils/param_limits.txt");
        const char hash = '#';
    const char endline = '\0';
    std::string var_name;
    while (!file.eof())
    {
        double vals[2];
        std::getline(file, line);
        if (line[0] == hash || line[0] == endline) continue;
        else
        {
            std::stringstream buffer(line);
            // Note: assumes tab or space separation
            buffer >> var_name;
            if (var_name.compare("tau_0") == 0)             buffer >> vals[0] >> vals[1];
            else if (var_name.compare("Lambda_0") == 0)     buffer >> vals[0] >> vals[1];
            else if (var_name.compare("xi_0") == 0)         buffer >> vals[0] >> vals[1];
            else if (var_name.compare("alpha_0") == 0)      buffer >> vals[0] >> vals[1];
            else if (var_name.compare("mass") == 0)         buffer >> vals[0] >> vals[1];
            else if (var_name.compare("eta_s") == 0)        buffer >> vals[0] >> vals[1];
            else if (var_name.compare("pi0") == 0)          buffer >> vals[0] >> vals[1];
            else if (var_name.compare("Pi0") == 0)          buffer >> vals[0] >> vals[1];
        } // end else
        _param_limits.insert(std::make_pair<std::string, std::vector<double>>(std::move(var_name), { vals[0], vals[1] }));
    } // end while(!fin.eof())
    }

    template<class HydroTheory, class...Args>
    double TestBayesianParameterEstimation<HydroTheory, Args...>::LogLikelihood(std::vector<double> tau_f, std::vector<double> observable, std::vector<double> observable_err, Args&&... args)
    {
        // TO DO: Rewrite...
        // Run simulation for soecific hydro model
        double log_likelihood = 1;
        HydroTheory copy = _model;
        for (size_t i = 0; i < tau_f.size(); i++)
        {
            _params.SetParameter("ul", tau_f[i]);
            copy.RunHydroSimulation(_params, std::forward<Args>(args)...);
            double model_observable = copy.e1;
            log_likelihood *= std::exp(-0.5 * std::pow(observable[i] - model_observable, 2.0) / std::pow(observable_err[i], 2.0)) / (std::sqrt(2.0 * PI) * observable_err[i]);
        }
        return std::log(log_likelihood);
    }

    template<class HydroTheory, class...Args>
    double TestBayesianParameterEstimation<HydroTheory, Args...>::LogPrior(void)
    {
        // For now, I just assume that everything has a uniform probability
        double prior = 1;
        for (auto param_limit : _param_limits)
            prior *= 1.0 / (param_limit.second[1] - param_limit.second[0]);

        return std::log(prior);
    }

    template<class HydroTheory, class...Args>
    double TestBayesianParameterEstimation<HydroTheory, Args...>::LogPosterior(double tau_f, double observable, double observable_err)
    {
        return LogLikelihood(tau_f, observable, observable_err) * LogPrior();
    }

    template<class HydroTheory, class...Args>
    void TestBayesianParameterEstimation<HydroTheory, Args...>::RunMCMC(int burnin, int steps, const char* out_filename, Args&&... args)
    {
        std::fstream fout(out_filename, std::fstream::out);
        if (!fout.is_open())
        {
            Print_Error(std::cerr, "output/bayes/ directory does not exist, please makes sure to create it firts.");
            exit(-1222);
        }
        double etas_lo = _param_limits["eta_s"][0];
        double etas_hi = _param_limits["eta_s"][1];
        
        double etas_step = (etas_hi - etas_lo) / (double) steps;

        _log_posterior.resize(steps);
#if USE_PARALLEL
        omp_set_dynamic(0);
        #pragma omp parallel for shared(_log_posterior) num_threads(4)
#endif
        for (int i = 0; i < steps; i++)
        {
            if (i % 100 == 0) Print(std::cout, fmt::format("n: {}", i));
            double etas = etas_lo + (double) i * etas_step;
            _params.SetParameter("eta_s", etas);
            
            int n = 2000;
            std::vector<double> sub_tauf(&_tau_fs[n], &_tau_fs[n + 10]);
            std::vector<double> sub_e(&_energy_densities[n], &_energy_densities[n+10]);
            std::vector<double> sub_e_err(&_energy_density_errors[n], &_energy_density_errors[n+10]);

            _log_posterior[i] = LogLikelihood(sub_tauf, sub_e, sub_e_err, std::forward<Args>(args)...) + LogPrior();
        }

        for (int i = 0; i < steps; i++)
            Print(fout, etas_lo + (double)i * etas_step, _log_posterior[i]);
        fout.close();
    }
}