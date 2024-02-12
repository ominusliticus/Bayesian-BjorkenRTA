#  Copyright 2021-2022 Kevin Ingles
#
#  Permission is hereby granted, free of charge, to any person obtaining
#  a copy of this software and associated documentation files (the
#  "Software"), to deal in the Software without restriction, including
#  without limitation the right to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Sofware is furnished to do so, subject to
#  the following conditions:
#
#  The above copyright notice and this permission notice shall be
#  included in all copies or substantial poritions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#  SOFTWARE OR THE USE OF OTHER DEALINGS IN THE SOFTWARE
#
# Author: Kevin Ingles
# File: HydroBayesianAnalysis.py
# Description: This file defines the Bayesian inference routines used for
#              parameter estimation and model comparison

# For changing directories to C++ programming and runnning files
from typing import Dict, List  # Dict and List args need to be added for all

# Typical functionality for data manipulation and generation of latin hypercube
import numpy as np
# import ptemcee
import my_ptemcee as ptemcee

# For plotting posteriors
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import my_plotting as mp

# For calculation
from scipy.linalg import lapack
from scipy.stats import dirichlet

# data storage
import pickle

# To create output directories if they don't already exist
from subprocess import run as cmd
from subprocess import CalledProcessError

# maximize processes used
from os import cpu_count

# Run MCMC calibration in parallel
from multiprocessing import Manager, Process
from multiprocessing import current_process

from typing import Tuple
from typing import Type
from typing import Optional


class HydroBayesianAnalysis(object):
    """
    This class is the API for all Bayesian Analysis methods, including
    parameter inference, model selection and model averaging.
    ----
    Constructor parameters:
    hydro_names      - list of hydros to run, either strings with names or numbers
    default_params   - parameter dictionary which stores the params we
                               wish to infer upon
    parameter_names  - the keys to the above dictionary (seems
                               redundant)
    parameter_ranges - the ranges to which each parameter is limited
    simulation_taus  - list of times to output the observable from
                               hydro codes
    """

    def __init__(
            self,
            hydro_names: List[str],
            default_params: Dict[str, float],
            parameter_names: List[str],
            parameter_ranges: np.ndarray,
            simulation_taus: np.ndarray,
            do_bmm: bool = False,
    ) -> None:
        print("Initializing Bayesian Analysis class")
        self.hydro_names = hydro_names
        self.params = default_params
        self.parameter_names = parameter_names
        self.num_params = len(parameter_names)
        self.parameter_ranges = parameter_ranges
        self.simulation_taus = simulation_taus
        self.do_bmm = do_bmm
        self.do_calibration_simultaneous = False

        self.MCMC_chains = None   # Dict[str, np.ndarray]
        self.evidence = None      # Dict[str, float]
        self.bmm_MCMC_chains = None

    def log_prior(self,
                  evaluation_point: np.ndarray,
                  parameter_ranges: np.ndarray) -> float:
        '''
        Parameters:
        ------------
        evaluation_points    - 1d-array with shape (n,m). \n
                               Value of parameters used to evaluate model \n
        design_range         - 2d-array with shape (m,2). \n
                               Give upper and lower limits of parameter values

        where m = number of parameters in inference
        '''
        X = np.array(evaluation_point).reshape(1, -1)
        if self.do_calibration_simultaneous:
            lower = np.all(X >= np.array(parameter_ranges)[:, 0])
            upper = np.all(X <= np.array(parameter_ranges)[:, 1])
        else:
            n_models = len(self.hydro_names)
            lower = np.all(X >= np.array(parameter_ranges)[:n_models, 0])
            upper = np.all(X <= np.array(parameter_ranges)[:n_models, 1])

        if (np.all(lower) and np.all(upper)):
            return 0
        else:
            return -np.inf

    def log_likelihood(self,
                       evaluation_point: np.ndarray,
                       true_observables: np.ndarray,
                       true_errors: np.ndarray,
                       hydro_name: str,
                       GP_emulator: Dict) -> np.ndarray:
        '''
        Parameters:
        ------------
        evaluation_points    - 1d-array like (1, num_params) \n
        true_observables     - data \n
        true_error           - data error \n
        hydro_name           - string containing hydro theory: 'ce', 'dnmr',
                                                               'vah', 'mvah'\n
        GP_emulator          - dictionary(hydro_name: emulator_list), \n
                                emulator_list[0] - energery density \n
                                emulator_list[1] - shear stress  \n
                                emulator_list[2] - bulk stress


        Returns:
        -----------
        Float - log-likelihood
        '''
        def predict_observable(evaluation_points: np.ndarray,
                               hydro_name: str,
                               tau_index: int,
                               GP_emulator: Dict) -> np.ndarray:
            """
            Function takes in the emulators list, and tau_index corresponding
            to the observation to be evaluated at and returns the predicted
            observables.

            Parameters:
            ------------
            evaluation_points - array of points from parameter space on which
                                to evaluate he likelihood (generally supplied
                                by an Monte Carlo Sampler)
            hydro_name        - name of hydro theory for which to run emulator
            tau_index         - index specifying the entry in
                                self.simulation_taus array to evaluate
                                emulator for

            Returns:
            -----------
            Tuple[emulator prediction, emulation error]
            """
            means = []
            variances = []
            for i in range(self.n_observables):
                prediction, error = \
                    GP_emulator[hydro_name][tau_index][i].predict(
                        np.array(evaluation_points).reshape(1, -1),
                        return_std=True)
                mean = prediction.reshape(-1, 1)
                std = error.reshape(-1,)

                means.append(mean)
                variances.append(std ** 2)
            return np.hstack(means), np.diag(np.array(variances).flatten())

        self.n_observables = true_observables.shape[-1]

        running_log_likelihood = [] if self.do_bmm else 0.0
        for k in range(true_observables.shape[0]):
            emulation_values, emulation_variance = \
                predict_observable(evaluation_point,
                                   hydro_name,
                                   k, GP_emulator)

            y = np.array(emulation_values).flatten() - \
                np.array(true_observables[k]).flatten()
            cov = emulation_variance + np.diag(true_errors[k].flatten()) ** 2

            # Use Cholesky decomposition for efficient lin alg algo
            L, info = lapack.dpotrf(cov, clean=True)

            if (info < 0):
                raise print(
                    'Error occured in computation of Cholesky decomposition')

            # Solve equation L*b=y
            b, info = lapack.dpotrs(L, np.array(y))

            if (info != 0):
                raise print('Error in inverting matrix equation')

            if np.all(L.diagonal() > 0):
                if self.do_bmm:
                    running_log_likelihood.append(-0.5 * np.dot(y, b) -
                                                  np.log(L.diagonal()).sum())
                else:
                    running_log_likelihood += -0.5 * np.dot(y, b) - \
                        np.log(L.diagonal()).sum()
            else:
                raise print('Diagonal has negative entry')

        return np.array(running_log_likelihood)

    def run_calibration(
            self,
            nsteps: int,
            nburn: int,
            ntemps: int,
            true_observables: np.ndarray,
            true_error: np.ndarray,
            GP_emulators: Dict,
            output_path: str,
            read_from_file: bool = False,
            run_parallel: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
        --------------
        nsteps : Number of steps for MCMC chain to take\n
        nburn : Number of burn-in steps to take\n
        ntemps : number of temperature for ptemcee sampler
        exact_observables : (n,4) np.ndarray where n is the number of\n
                            simulation_taus passed at initialization\n
        exact_error : (n,3) np.ndarray where n is the number of\n
                      simulation_taus passed at initialization
        GP_emulators : Dictionary of list of emulators
        output_path : Define path where to output mcmc chains to load previous
                      runs
        read_from_file : Boolean, read last run only works if existing run
                         exists.

        Returns:
        ----------
        Dictionary of MCMC chains, index by the hydro name.\n
        MCMC chain has the shape (ntemps, nwalkers, nsteps, num_params),
        where nwalkers = 20 * num_params
        """

        self.n_observables = true_observables.shape[-1]      
        if read_from_file:
            print("Reading mcmc_chain from file")
            with open(f'{output_path}/mcmc_chains.pkl',
                      'rb') as f:
                self.MCMC_chains = pickle.load(f)
            return self.MCMC_chains
        else:
            nwalkers = 20 * self.num_params
            
            def for_multiprocessing(
                    hydro_name: str,
                    output_dict: Type[Dict[str, np.ndarray]],
                    itr: Optional[int] = None,
            ):
                starting_guess = np.array(
                    [self.parameter_ranges[:, 0] +
                    np.random.rand(nwalkers, self.num_params) *
                    np.diff(self.parameter_ranges).reshape(-1,)
                    for _ in range(ntemps)])
                sampler = ptemcee.Sampler(nwalkers=nwalkers,
                                        dim=self.num_params,
                                        ntemps=ntemps,
                                        Tmax=1000,
                                        threads=cpu_count(),
                                        logl=self.log_likelihood,
                                        logp=self.log_prior,
                                        loglargs=[true_observables[:, 1:4],
                                                    true_error,
                                                    hydro_name,
                                                    GP_emulators],
                                        logpargs=[self.parameter_ranges])
                
                if itr is None:
                    desc = None
                else:
                    desc = hydro_name
                position = itr

                x = sampler.run_mcmc(p0=starting_guess,
                                    iterations=nburn,
                                    swap_ratios=True,
                                    desc=desc,
                                    position=position,)
                sampler.reset()
                x = sampler.run_mcmc(p0=x[0],
                                    iterations=nsteps,
                                    storechain=True,
                                    swap_ratios=True,
                                    desc=desc,
                                    position=position,)

                output_dict[hydro_name] = np.array(sampler.chain)

            if run_parallel:
                manager = Manager()
                self.MCMC_chains = manager.dict()
                for name in self.hydro_names:
                    self.MCMC_chains[name] = None

                jobs = [Process(target=for_multiprocessing,
                                args=(name,
                                      self.MCMC_chains,
                                      i))
                        for i, name in enumerate(self.hydro_names)]
                _ = [job.start() for job in jobs]
                _ = [job.join() for job in jobs]

                self.MCMC_chains = dict(self.MCMC_chains)                    
            else:
                self.MCMC_chains = {}
                for i, name in enumerate(self.hydro_names):
                    for_multiprocessing(name, self.MCMC_chains)

            try:
                (cmd(['mkdir', '-p', f'{output_path}'])
                    .check_returncode())
            except (CalledProcessError):
                print(f"Could not create dir {output_path}")

            with open(f'{output_path}/mcmc_chains.pkl',
                      'wb') as f:
                pickle.dump(self.MCMC_chains, f)

            return self.MCMC_chains

    def calculate_bayes_factor(self, hydro1: str, hydro2: str) -> float:
        """
        Parameters:
        -------------
        hydro1: name of 1 of four hydro theory being compared
        hydro2: name of a different hydro theory being compared

        Returns:
        -------------
        The Bayes factor, or ratio of evidences, for hydro1 to hydro2
        """
        return self.evidence[hydro1][0] / self.evidence[hydro2][0]

    def plot_posteriors(self, output_dir: str, axis_names: List[str]):
        if not self.do_calibration_simultaneous and self.MCMC_chains is None:
            print("Reading mcmc_chain from file")
            with open(f'{output_dir}/mcmc_chains.pkl',
                      'rb') as f:
                self.MCMC_chains = pickle.load(f)

        if self.MCMC_chains is None:
            if self.do_bmm and self.bmm_MCMC_chains is None:
                print("No chains to plot. Please run calibration")
                return

        # TODO: Add true and MAP values to plot
        # pallette = sns.color_palette('Colorblind')
        
        weights_offset = len(self.hydro_names)
        if self.do_calibration_simultaneous:
            data = self.bmm_MCMC_chains[0, ..., weights_offset:].reshape(
                -1,
                len(self.parameter_names)
            )
            
            df = pd.DataFrame(dict((name, data[:, k])
                                   for k, name in enumerate(axis_names)))
            g = sns.pairplot(
                data=df,
                corner=True,
                diag_kind='kde',
                kind='hist'
            )
            g.map_lower(sns.kdeplot, levels=4, color='black')
            g.tight_layout()
            g.savefig('{}/plots/bmm_simultaneous_corner_plot_n={}.pdf'.
                        format(output_dir, self.num_params))

        else:
            dfs = pd.DataFrame(columns=[*axis_names, 'hydro'])
            for i, name in enumerate(self.hydro_names):
                data = self.MCMC_chains[name][0].reshape(
                    -1,
                    len(self.parameter_names)
                )
                df = pd.DataFrame(dict((name, data[:, k])
                                        for k, name in enumerate(axis_names)))
                g1 = sns.pairplot(
                    data=df,
                    corner=True,
                    diag_kind='kde',
                    kind='hist'
                )
                g1.map_lower(sns.kdeplot, levels=4, color='black')
                g1.tight_layout()
                g1.savefig('{}/plots/{}_corner_plot_n={}.pdf'.
                            format(output_dir, name, self.num_params))

                df['hydro'] = name
                dfs = pd.concat([dfs, df], ignore_index=True)

            g = sns.pairplot(data=dfs,
                             corner=True,
                             diag_kind='kde',
                             kind='hist',
                             hue='hydro')
            g.legend()
            g.map_lower(sns.kdeplot, levels=4, color='black')
            g.tight_layout()

            try:
                (cmd(['mkdir', '-p', f'{output_dir}/plots'])
                    .check_returncode())
            except (CalledProcessError):
                print(f"Could not create dir {output_dir}/plots")
            g.savefig(
                f'{output_dir}/plots/all_corner_plot_n={self.num_params}.pdf')

    def plot_weights(self, output_dir: str, use_median: bool = False) -> None:
        """
        Plot the posterior predictive distributions of the weights

        Parameters:
        ===========
        output_dir: str
            path to where the MCMC runs are stored, and corresponding plots
            are to be outputed
        use_median: bool (default: False)
            Calculate median (and quartiles), as opposed to mean (and 
            standard deviation), for the posteriors
        """
        # TODO: We could make these density plots on log scale instead
        if self.bmm_MCMC_chains is not None:
            n_models = len(self.hydro_names)
            weights = self.weights[0].reshape(
                -1,
                *self.weights.shape[-2:],
            )
            n_observation = weights.shape[-1]
            assert n_observation == self.simulation_taus.shape[0]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
            fig.patch.set_facecolor('white')
            cmap = mp.get_cmap(10, 'tab10')
            for i in range(n_models):
                if use_median:
                    q1, median, q2 = np.quantile(
                        a=weights[:, i],
                        q=(0.25, 0.50, 0.75)
                    )
                    ax.plot(
                        self.simulation_taus,
                        median,
                        lw=2,
                        color=cmap(i),
                        label=self.hydro_names[i]
                    )
                    ax.fill_between(
                        self.self.simulation_taus,
                        q1,
                        q2,
                        color=cmap(i),
                        alpha=0.5
                    )
                else:
                    mean = np.mean(weights[:, i], axis=0)
                    std = np.std(weights[:, i], axis=0)
                    ax.plot(
                        self.simulation_taus,
                        mean,
                        lw=2,
                        color=cmap(i),
                        label=self.hydro_names[i]
                    )
                    ax.fill_between(
                        self.simulation_taus,
                        mean + std,
                        mean - std,
                        color=cmap(i),
                        alpha=0.5,
                    )
            mp.costumize_axis(ax, r'$\tau$ [fm/c]', r'$w(\tau)$')
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=20)

            try:
                (cmd(['mkdir', '-p', f'{output_dir}/plots'])
                    .check_returncode())
            except (CalledProcessError):
                print(f"Could not create dir {output_dir}/plots")
            fig.savefig(
                f'{output_dir}/plots/bmm_weights_n={self.num_params}.pdf')


    # Expand to included Bayesian Model mixing for paper (will migrate
    # everything to Taweret later)

    def get_weights(
            self,
            evaluation_point: np.ndarray,
            x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculates the position-dependent weights with missing method 
        `self.missing method`.
        Supported mixing methods include:
            - dirichlet
            - gauss_process
        Defaults to Dirichlet mixing 

        Parameters:
        ===========
        evaluation_point: np.ndarray
            The array containing the sampled parameters for the inferences
        
        x: Optional[np.ndarray] (default: None)
            Location to evaluation weight
        """
        if self.mixing_method == 'dirichlet':
            # log_ws will have shape (m_models, n_observations) after
            # transpose
            return np.log(
                dirichlet(
                    np.abs(1 / evaluation_point[:self.num_models])
                ).rvs(size=self.true_observables.shape[0])).T
        elif self.mixing_method == 'gauss_process':
            pass
        else:
            print('mixing method not supported')


    def mixing_log_likelihood(
        self,
        evaluation_point: np.ndarray,
        true_observables: np.ndarray,
        true_errors: np.ndarray,
        hydro_names: List[str],
        GP_emulator: Dict,
        fixed_evaluation_parameters_for_models: Dict[str, np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Mixing mixing likelihood that combines the likelhood distributions for
        the models.

        Parameters:
        -----------
        evaluation_points    - 1d-array like (1, num_params + num_models) \n
        true_observables     - data should have shape
                               (n_observations, n_observables) \n
        true_error           - data error should have shape
                               (n_observations, n_observables - 1) \n
        hydro_names          - list containing names of hydro being compared
        GP_emulator          - dictionary(hydro_name: emulator_list), \n
                                emulator_list[0] - energery density \n
                                emulator_list[1] - shear stress  \n
                                emulator_list[2] - bulk stress


        Returns:
        -----------
        Float - mixing log-likelihood, mixing weights
        '''
        self.num_models = len(hydro_names)
        self.true_observables = true_observables
        # log_ws will have shape (m_models, n_observation_times) after
        # transpose
        log_ws = self.get_weights(
            evaluation_point=evaluation_point,
            x=true_observables[:, 0]
        )

        if fixed_evaluation_parameters_for_models is None:
            # log_likelihood will return an array of shape (n_observation_times,),
            # therefore, model_log_likelihoods should have shape
            # (n_models, n_observation_times)
            model_log_likelihoods = np.array([
                self.log_likelihood(
                    evaluation_point=evaluation_point[self.num_models:],
                    true_observables=true_observables,
                    true_errors=true_errors,
                    hydro_name=hydro_name,
                    GP_emulator=GP_emulator
                )
                for hydro_name in hydro_names
            ])
        else:
            # This should take the previous posteriors, not just the MAP values
            model_log_likelihoods = np.array([
                self.log_likelihood(
                    evaluation_point=fixed_evaluation_parameters_for_models[
                        hydro_name
                    ],
                    true_observables=true_observables,
                    true_errors=true_errors,
                    hydro_name=hydro_name,
                    GP_emulator=GP_emulator
                )
                for hydro_name in hydro_names
            ])

        # The sampler is coded such that it expects the weights to have the shape
        # (n_models, n_observation_times)
        ll = np.prod(np.logaddexp.reduce(
            log_ws + model_log_likelihoods, axis=1))
        ws = np.exp(log_ws)
        # print("From inside log_likelihood", ll, ws)
        return (ll, ws)

    def run_mixing(
        self,
        nsteps: int,
        nburn: int,
        ntemps: int,
        exact_observables: np.ndarray,
        exact_error: np.ndarray,
        GP_emulators: Dict,
        output_path: str,
        do_calibration_simultaneous: bool = False,
        fixed_evaluation_points_models: Dict[str, np.ndarray] = None,
        read_from_file: bool = False,
    ) -> np.ndarray:
        '''
        Parameters:
        --------------
        nsteps : Number of steps for MCMC chain to take\n
        nburn : Number of burn-in steps to take\n
        ntemps : number of temperature for ptemcee sampler
        exact_observables : (n,4) np.ndarray where n is the number of\n
                            simulation_taus passed at initialization\n
        exact_error : (n,3) np.ndarray where n is the number of\n
                      simulation_taus passed at initialization
        GP_emulators : Dictionary of list of emulators
        output_path : Define path where to output mcmc chains to load previous
                      runs
        read_from_file : Boolean, read last run only works if existing run
                         exists.

        Returns:
        ----------
        Dictionary of MCMC chains, index by the hydro name.\n
        MCMC chain has the shape:
            (ntemps, nwalkers, nsteps,num_params + n_models)
        where nwalkers = 20 * num_params
        '''
        self.do_calibration_simultaneous = do_calibration_simultaneous
        if read_from_file:
            print("Reading bmm_mcmc_chain from file")
            with open(f'{output_path}/bmm_mcmc_chains.pkl',
                      'rb') as f:
                self.bmm_MCMC_chains = pickle.load(f)
            with open(f'{output_path}/bmm_mcmc_weights.pkl',
                      'rb') as f:
                self.weights = pickle.load(f)
            
            return self.bmm_MCMC_chains, self.weights
        else:

            nwalkers = 20 * self.num_params
            n_models = len(self.hydro_names)

            # manager = Manager()
            # sampler = manager.dict()
            # for name in hydro_names:
            #     sampler[name] = []

            # def for_multiprocessing(sampler: Dict[str, float], name: str, **kwargs):
            #     sampler[name] = ptemcee.Sampler(**kwargs)

            if do_calibration_simultaneous:
                starting_guess = np.array(
                    [self.parameter_ranges[:, 0] +
                        np.random.rand(
                            nwalkers,
                            self.num_params + n_models
                    ) * np.diff(self.parameter_ranges).reshape(-1,)
                        for _ in range(ntemps)])
            else:
                # only need to initialize walkers for the weights
                starting_guess = np.array(
                    [self.parameter_ranges[:n_models, 0] +
                        np.random.rand(
                            nwalkers,
                            n_models
                    ) *
                        np.diff(self.parameter_ranges[:n_models])
                        .reshape(-1,)
                        for _ in range(ntemps)])

            # TODO: For the case where the fixed evaluation points are not
            #       given, run the calibration and extract the MAP values
            #       from the posteriors

            sampler = ptemcee.Sampler(
                nwalkers=nwalkers,
                dim=(self.num_params + n_models
                     if do_calibration_simultaneous else
                     n_models),
                ntemps=ntemps, Tmax=1000,
                threads=cpu_count(),
                logl=self.mixing_log_likelihood,
                logp=self.log_prior,
                loglargs=[exact_observables[:, 1:4],
                          exact_error,
                          self.hydro_names,
                          GP_emulators,
                          fixed_evaluation_points_models],
                logpargs=([self.parameter_ranges]
                          if do_calibration_simultaneous else
                          [self.parameter_ranges[:n_models]]),
                do_bmm=True,
                num_models=n_models,
                num_observations=exact_observables.shape[0]
            )
            print('burn in sampling started')
            x = sampler.run_mcmc(p0=starting_guess,
                                 iterations=nburn,
                                 swap_ratios=True)
            print('Burn in completed.')

            sampler.reset()

            print("Now running the samples")
            x = sampler.run_mcmc(p0=x[0],
                                 iterations=nsteps,
                                 storechain=True,
                                 swap_ratios=True)

            self.bmm_MCMC_chains = sampler.chain
            self.weights = sampler.weights

            try:
                (cmd(['mkdir', '-p', f'{output_path}'])
                    .check_returncode())
            except (CalledProcessError):
                print(f"Could not create dir {output_path}")

            with open(f'{output_path}/bmm_mcmc_chains.pkl',
                      'wb') as f:
                pickle.dump(self.bmm_MCMC_chains, f)

            with open(f'{output_path}/bmm_mcmc_weights.pkl',
                      'wb') as f:
                pickle.dump(self.weights, f)

            return self.bmm_MCMC_chains, self.weights
