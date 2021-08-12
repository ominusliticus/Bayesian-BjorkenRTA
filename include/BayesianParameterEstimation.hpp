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
    using data = std::vector<double>;

    template<class HydroTheory, class...Args>
    class TestBayesianParameterEstimation
    {
    public:
        TestBayesianParameterEstimation() = default;
        // Takes in vectors of data for calculating likelihood
        TestBayesianParameterEstimation(SP& params, data tau, data e, data e_err, data pi, data pi_err, data Pi, data Pi_err, HydroTheory& hydro);
        //TO DO: Add destrucors to clear up memory
        ~TestBayesianParameterEstimation();

        double LogLikelihood(Args&&... args);
        double LogPrior(void);
        double LogPosterior(double tau_f, double observable, double observable_err);
        void RunMCMC(int burnin, int steps, const char* out_filename, Args&&... args);

    protected:
        SP& _params;
        data _tau_fs;
        data _energy_densities;
        data _shear_pressures;
        data _bulk_pressures;
        data _energy_density_errors;
        data _shear_pressure_errors;
        data _bulk_pressure_errors;
        std::unordered_map<std::string, data> _param_limits;

        HydroTheory _model;
        data _log_posterior;
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
    TestBayesianParameterEstimation<HydroTheory, Args...>::TestBayesianParameterEstimation(SP& params, data tau, data e, data e_err, data pi, data pi_err, data Pi, data Pi_err, HydroTheory& hydro)
        : _params {params}, _tau_fs{tau}, _energy_densities {e},  _shear_pressures {pi}, _bulk_pressures {Pi}, _energy_density_errors {e_err}, _shear_pressure_errors {pi_err}, _bulk_pressure_errors {Pi_err}
    {
        _model = hydro;


        std::fstream file = std::fstream("utils/param_limits.txt");
        std::string line;
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
    TestBayesianParameterEstimation<HydroTheory, Args...>::~TestBayesianParameterEstimation()
    {
        // _model.~HydroModel();
        _tau_fs.clear();
        _energy_densities.clear();
        _shear_pressures.clear();
        _bulk_pressures.clear();
        _energy_density_errors.clear();
        _shear_pressure_errors.clear();
        _bulk_pressure_errors.clear();
        _param_limits.clear();
        _log_posterior.clear();
    }

    template<class HydroTheory, class...Args>
    double TestBayesianParameterEstimation<HydroTheory, Args...>::LogLikelihood(Args&&... args)
    {
        // TO DO: Switch to store output value: run to latest time and then interpolate results for previous times  
        // Run simulation for soecific hydro model
        double log_likelihood = 1;
        HydroTheory copy = _model;
        for (size_t i = 0; i < _tau_fs.size(); i++)
        {
            _params.SetParameter("ul", _tau_fs[i]);
            copy.RunHydroSimulation(_params, std::forward<Args>(args)...);
            double model_energy_density = copy.e1;
            double model_shear_pressure = copy.pi1;
            double model_bulk_pressure = copy.Pi1;
            log_likelihood *= std::exp(-0.5 * std::pow(_energy_densities[i] - model_energy_density, 2.0) / std::pow(_energy_density_errors[i], 2.0)) / (std::sqrt(2.0 * PI) * _energy_density_errors[i]);
            log_likelihood *= std::exp(-0.5 * std::pow(_shear_pressures[i] - model_shear_pressure, 2.0) / std::pow(_shear_pressure_errors[i], 2.0)) / (std::sqrt(2.0 * PI) * _shear_pressure_errors[i]);
            log_likelihood *= std::exp(-0.5 * std::pow(_bulk_pressures[i] - model_bulk_pressure, 2.0) / std::pow(_bulk_pressure_errors[i], 2.0)) / (std::sqrt(2.0 * PI) * _bulk_pressure_errors[i]);
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
            if (i % 10 == 0) Print(std::cout, fmt::format("n: {}", i));
            double etas = etas_lo + (double) i * etas_step;
            _params.SetParameter("eta_s", etas);

            _log_posterior[i] = LogLikelihood(std::forward<Args>(args)...) + LogPrior();
        }

        for (int i = 0; i < steps; i++)
            Print(fout, etas_lo + (double)i * etas_step, _log_posterior[i]);
        fout.close();
    }
}