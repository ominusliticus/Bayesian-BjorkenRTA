#
# Author: Kevin Ingles
# File: HydroBayesianAnalysis.py
# Description: This file defines the Bayesian inference routines used for 
#              parameter estimation and model comparison

# For changing directories to C++ programming and runnning files
from typing import Dict, List

# Typical functionality for data manipulation and generation of latin hypercube
import numpy as np
import ptemcee


# For calculations
from scipy.linalg import lapack

# data storage
import pickle

# TODO: 1. Separate Bayesian inference portion of code to separate step
#       2. Create function that inverts energy density, shear and bulk
#          pressure to give energy density, transverse and longitudinal
#          pressure
#       3. Separate file I/O from class and make separate routines
#       5. Reduce cyclomatic complexity


class HydroBayesianAnalysis(object):
    """
    Add description
    """
    def __init__(
            self,
            default_params: Dict,
            parameter_names: List,
            parameter_ranges: np.ndarray,
            simulation_taus: np.ndarray,
            run_new_hydro: bool,
            train_GP: bool
            ) -> None:
        print("Initializing Bayesian Analysis class")
        self.hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
        self.params = default_params
        self.parameter_names = parameter_names
        self.num_params = len(parameter_names)
        self.parameter_ranges = parameter_ranges
        self.simulation_taus = simulation_taus
        self.train_GP = train_GP
        self.run_new_hydro = run_new_hydro

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
                      GP_emulator: Dict,
                      scalers: Dict = {}) -> np.ndarray:
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
        '''
        def PredictObservable(evaluation_points: np.ndarray,
                              hydro_name: str,
                              tau_index: int,
                              GP_emulator: Dict,
                              scalers: Dict = {}) -> np.ndarray:
            """
            Add description
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
                               k, GP_emulator, scalers)

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
                read_from_file: bool = False) -> Dict:
        """
        Parameters:
        --------------
        nsteps : Number of steps for MCMC chain to take\n
        nburn : Number of burn-in steps to take\n
        exact_observables : (n,4) np.ndarray where n is the number of\n
                            simulation_taus passed at initialization\n
        exact_error : (n,3) np.ndarray where n is the number of\n
                      simulation_taus passed at initialization

        Returns:
        ----------
        Dictionary of MCMC chains, index by the hydro name.\n
        MCMC chain has the shape (ntemps, nwalkers, nsteps, num_params),
        where nwalkers = 20 * num_params
        """
        if read_from_file:
            print("Reading mcmc_chain from file")
            with open('pickle_files/mcmc_chains.pkl', 'rb') as f:
                self.MCMC_chains = pickle.load(f)
            with open('pickle_files/evidence.pkl', 'rb') as f:
                self.evidence = pickle.load(f)
        else:
            nwalkers = 20 * self.num_params

            # TO DO: print MCMC to pickle file
            for i, name in enumerate(self.hydro_names):
                print(f"Computing for hydro theory: {name}")
                starting_guess = np.array(
                    [self.parameter_ranges[:, 0] +
                     np.random.rand(nwalkers, self.num_params) *
                     np.diff(self.parameter_ranges)
                     for _ in range(ntemps)])
                sampler = ptemcee.Sampler(nwalkers=nwalkers,
                                          dim=self.num_params,
                                          ntemps=ntemps,
                                          threads=4,
                                          logl=self.LogLikelihood,
                                          logp=self.LogPrior,
                                          loglargs=[exact_observables[:, 1:4],
                                                    exact_error,
                                                    name,
                                                    self.GP_emulators,
                                                    self.scalers],
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

            with open('pickle_files/mcmc_chains.pkl', 'wb') as f:
                pickle.dump(self.MCMC_chains, f)
            with open('pickle_files/evidence.pkl', 'wb') as f:
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
