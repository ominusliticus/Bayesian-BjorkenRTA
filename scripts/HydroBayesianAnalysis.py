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
import ptemcee

# For plotting posteriors
import seaborn as sns
import pandas as pd

# For calculation
from scipy.linalg import lapack

# data storage
import pickle

# To create output directories if they don't already exist
from subprocess import run as cmd
from subprocess import CalledProcessError


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
            ) -> None:
        print("Initializing Bayesian Analysis class")
        self.hydro_names = hydro_names
        self.params = default_params
        self.parameter_names = parameter_names
        self.num_params = len(parameter_names)
        self.parameter_ranges = parameter_ranges
        self.simulation_taus = simulation_taus

        self.MCMC_chains = {}   # Dict[str, np.ndarray]
        self.evidence = {}      # Dict[str, float]

    def LogPrior(self,
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
        lower = np.all(X >= np.array(parameter_ranges)[:, 0])
        upper = np.all(X <= np.array(parameter_ranges)[:, 1])

        if (np.all(lower) and np.all(upper)):
            return 0
        else:
            return -np.inf

    def LogLikelihood(self,
                      evaluation_point: np.ndarray,
                      true_observables: np.ndarray,
                      true_errors: np.ndarray,
                      hydro_name: str,
                      GP_emulator: Dict) -> np.ndarray:
        '''
        Parameters:
        ------------
        evaluation_points    - 1d-array like (1, num_params) \n
        true_observables     - data
        true_error           - data error
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
        def PredictObservable(evaluation_points: np.ndarray,
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
            for i in range(3):
                prediction, error = \
                 GP_emulator[hydro_name][tau_index][i].predict(
                    np.array(evaluation_points).reshape(1, -1),
                    return_std=True)
                mean = prediction.reshape(-1, 1)
                std = error.reshape(-1,)

                means.append(mean)
                variances.append(std ** 2)
            return np.hstack(means), np.diag(np.array(variances).flatten())

        running_log_likelihood = 0
        for k in range(true_observables.shape[0]):
            emulation_values, emulation_variance = \
                 PredictObservable(evaluation_point,
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
                running_log_likelihood += -0.5 * np.dot(y, b) - \
                    np.log(L.diagonal()).sum()
            else:
                raise print('Diagonal has negative entry')

        return running_log_likelihood

    def RunMCMC(self,
                nsteps: int,
                nburn: int,
                ntemps: int,
                exact_observables: np.ndarray,
                exact_error: np.ndarray,
                GP_emulators: Dict,
                output_path: str,
                read_from_file: bool = False) -> Dict:
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
        if read_from_file:
            print("Reading mcmc_chain from file")
            with open(f'{output_path}/pickle_files/mcmc_chains.pkl',
                      'rb') as f:
                self.MCMC_chains = pickle.load(f)
            with open(f'{output_path}/pickle_files/evidence.pkl', 'rb') as f:
                self.evidence = pickle.load(f)
        else:
            nwalkers = 20 * self.num_params

            for i, name in enumerate(self.hydro_names):
                print(f"Computing for hydro theory: {name}")
                starting_guess = np.array(
                    [self.parameter_ranges[:, 0] +
                     np.random.rand(nwalkers, self.num_params) *
                     np.diff(self.parameter_ranges).reshape(-1,)
                     for _ in range(ntemps)])
                sampler = ptemcee.Sampler(nwalkers=nwalkers,
                                          dim=self.num_params,
                                          ntemps=ntemps,
                                          Tmax=10,
                                          threads=4,
                                          logl=self.LogLikelihood,
                                          logp=self.LogPrior,
                                          loglargs=[exact_observables[:, 1:4],
                                                    exact_error,
                                                    name,
                                                    GP_emulators],
                                          logpargs=[self.parameter_ranges])

                print('burn in sampling started')
                x = sampler.run_mcmc(p0=starting_guess,
                                     iterations=nburn,
                                     swap_ratios=True)
                print("Mean acceptance fractions (in total {0} steps): "
                      .format(ntemps*nwalkers*nburn))
                print(x[3])
                print('Burn in completed.')

                sampler.reset()

                print("Now running the samples")
                x = sampler.run_mcmc(p0=x[0],
                                     iterations=nsteps,
                                     storechain=True,
                                     swap_ratios=True)
                print("Mean acceptance fractions (in total {0} steps): "
                      .format(ntemps * nwalkers * nsteps))
                print(x[3])

                self.MCMC_chains[name] = np.array(sampler.chain)
                self.evidence[name] = sampler.log_evidence_estimate()

            try:
                (cmd(['mkdir', '-p', f'{output_path}/pickle_files'])
                    .check_returncode())
            except (CalledProcessError):
                print(f"Could not create dir {output_path}/pickle_files")

            with open(f'{output_path}/pickle_files/mcmc_chains.pkl',
                      'wb') as f:
                pickle.dump(self.MCMC_chains, f)
            with open(f'{output_path}/pickle_files/evidence.pkl', 'wb') as f:
                pickle.dump(self.evidence, f)

    def CalculateBayesFactor(self, hydro1: str, hydro2: str) -> float:
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

    def PlotPosteriors(self, output_dir: str, axis_names: List[str]):
        # TODO: Add true and MAP values to plot
        dfs = pd.DataFrame(columns=[*axis_names, 'hydro'])
        # pallette = sns.color_palette('Colorblind')
        for i, name in enumerate(self.hydro_names):
            data = self.MCMC_chains[name][0].reshape(-1,
                                                     len(self.
                                                         parameter_names))
            df = pd.DataFrame(dict((name, data[:, i])
                              for i, name in enumerate(axis_names)))
            g1 = sns.pairplot(data=df,
                              corner=True,
                              diag_kind='kde',
                              kind='hist')
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
        g.map_lower(sns.kdeplot, levels=4, color='black')
        g.tight_layout()
        g.savefig(
            f'{output_dir}/plots/all_corner_plot_n={self.num_params}.pdf')
