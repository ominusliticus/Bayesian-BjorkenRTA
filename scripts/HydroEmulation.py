#
# Author: Kevin Ingles
# File: HydroEmulation.py
# Description: This file constructs the Gaussian emulators from hydro
#              simulations for efficient code evaluation

# For type identification
from typing import List, Dict
import numpy as np

# For choosing training data
from pyDOE import lhs

# Gaussian Process emulator
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

# Serializing python objects
import pickle

# To interface with Hydro code
from HydroCodeAPI import HydroCodeAPI

# For storing data
import pandas as pd

# For plotting residuals
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


# TOOD: Add workflow that systematically tests the emulator
#       preformance by having a training and testing set of hydro
#       and quantifying how well the emulator reproduces the value
class HydroEmulator:
    """
    Add description
    """

    def __init__(self,
                 hca: HydroCodeAPI,
                 params_dict: Dict[str, float],
                 parameter_names: List[str],
                 parameter_ranges: np.ndarray,
                 simulation_taus: np.ndarray,
                 hydro_names: List[str],
                 use_existing_emulators: bool,
                 use_PT_PL: bool):
        '''
        Add description
        '''
        # TODO: If hydro output contains NaN, output the design point
        #       for which the NaN(s) were present
        self.GP_emulators = dict((key, None) for key in hydro_names)
        self.b_use_existing_emulators = use_existing_emulators
        if use_existing_emulators:
            # Load GP data from pickle files
            with open(
                    'design_points/design_points_n={}.dat'.
                    format(len(parameter_names)),
                    'r'
                    ) as f:
                self.design_points = np.array(
                    [[float(entry) for entry in line.split()]
                     for line in f.readlines()])

            f_pickle_emulators = open(
                'pickle_files/emulators_data_n={}.pkl'.
                format(len(parameter_names)),
                'rb')

            self.GP_emulators = pickle.load(f_pickle_emulators)
            f_pickle_emulators.close()
        else:
            print("Running hydro")
            # Run hydro code and generate scalers and GP pickle files
            unit = lhs(n=len(parameter_names),
                       # add 10 points for testing data
                       samples=20 * len(parameter_names) + 10,
                       criterion='maximin')
            self.design_points = parameter_ranges[:, 0] + unit * \
                (parameter_ranges[:, 1] - parameter_ranges[:, 0])

            design_points = self.design_points[:-10]
            self.test_points = self.design_points[-10:]
            with open('design_points/design_points_n={}.dat'.
                      format(len(parameter_names)), 'w') as f:
                for line in design_points.reshape(-1, len(parameter_names)):
                    for entry in line:
                        f.write(f'{entry} ')
                    f.write('\n')
            with open('design_points/testing_points_n={}.dat'.
                      format(len(parameter_names)), 'w') as f:
                for line in self.test_points.reshape(-1, len(parameter_names)):
                    for entry in line:
                        f.write(f'{entry} ')
                    f.write('\n')

            hca.RunHydro(params_dict=params_dict,
                         parameter_names=parameter_names,
                         design_points=design_points,
                         simulation_taus=simulation_taus,
                         use_PT_PL=use_PT_PL)

            hydro_simulations = dict((key, []) for key in hydro_names)
            for k, name in enumerate(hydro_names):
                for j, tau in enumerate(simulation_taus):
                    with open(('hydro_simulation_points/{}_simulation_points'
                               + '_n={}_tau={}.dat').
                              format(name, len(parameter_names), tau),
                              'r') as f_hydro_simulation_pts:
                        hydro_simulations[name].append(
                            [[float(entry)
                              for entry in line.split()]
                             for line in f_hydro_simulation_pts.readlines()
                             ])
            hydro_simulations = dict(
                (key, np.array(hydro_simulations[key]))
                for key in hydro_simulations)

            print("From fitting emulators: ",
                  hydro_simulations['ce'].shape)
            print("Fitting emulators")
            hydro_lists = np.array(
                [hydro_simulations[key] for key in hydro_names])

            self.GP_emulators = dict((key, []) for key in hydro_names)

            obs = ['E', 'P1', 'P2']
            f_emulator_scores = open(
                f'full_outputs/emulator_scores_n={len(parameter_names)}.txt',
                'w')
            f_pickle_emulators = open(
                f'pickle_files/emulators_data_n={len(parameter_names)}.pkl',
                'wb')

            for i, name in enumerate(hydro_names):
                global_emulators = []
                for j, tau in enumerate(simulation_taus):
                    local_emulators = []
                    f_emulator_scores.write(f'\tTraining GP for {name}\n')
                    for m in range(1, 4):
                        data = hydro_lists[i, j, :, m].reshape(-1, 1)

                        bounds = np.outer(
                            np.diff(parameter_ranges), (1e-2, 1e2))
                        kernel = 1 * krnl.RBF(
                                length_scale=np.diff(parameter_ranges),
                                length_scale_bounds=bounds)
                        GPR = gpr(kernel=kernel,
                                  n_restarts_optimizer=40,
                                  alpha=1e-8,
                                  normalize_y=True)
                        f_emulator_scores.write(
                            f'\t\tTraining GP for {name} and time {tau}\n')
                        try:
                            GPR.fit(
                                design_points.reshape(-1,
                                                      len(parameter_names)),
                                data)
                        except ValueError:
                            print("NaN encountered for design point:\n{}\n{}".
                                  format(design_points, data))
                            print("Error occured in iteration ({},{},{})".
                                  format(i, j, m))

                        f_emulator_scores.write(
                            f'''Runnig fit for {name} at time {tau} fm/c
                                for observable {obs[m-1]}''')
                        f_emulator_scores.write(
                            'GP score: {:1.3f}'.format(
                                GPR.score(
                                    design_points.reshape(-1,
                                                          len(parameter_names)
                                                          ),
                                    data))
                            )
                        f_emulator_scores.write(
                            'GP parameters: {}'.format(GPR.kernel_))
                        f_emulator_scores.write(
                            'GP log-likelihood: {}'.format(
                                GPR.log_marginal_likelihood(GPR.kernel_.theta))
                            )
                        f_emulator_scores.write(
                            '------------------------------\n')
                        local_emulators.append(GPR)
                    global_emulators.append(local_emulators)
                self.GP_emulators[name] = global_emulators

            pickle.dump(self.GP_emulators, f_pickle_emulators)
            f_emulator_scores.close()

            print("Done")

    def TestEmulator(self,
                     hca: HydroCodeAPI,
                     params_dict: Dict[str, float],
                     parameter_names: List[str],
                     parameter_ranges: np.ndarray,
                     simulation_taus: np.ndarray,
                     hydro_names: List[str],
                     use_existing_emulators: bool,
                     use_PT_PL: bool) -> None:
        '''
        This function takes a given set of emulators and tests
        them for how accurately they run
        Parameters:
        ----------
        type annotations are specific enough

        Returns:
        ----------
        None
        '''
        if self.b_use_existing_emulators:
            with open('design_points/testing_points_n={}.dat'.
                      format(len(parameter_names)), 'r') as f:
                self.test_points = np.array(
                    [[float(entry)
                      for entry in line]
                     for line in f.readlines()]
                )
            with open('pickle_files/emulator_testing_data.pkl', 'rb') as f:
                hydro_simulations = pickle.load(f)
        else:
            hca.RunHydro(params_dict=params_dict,
                         parameter_names=parameter_names,
                         design_points=self.test_points,
                         simulation_taus=simulation_taus,
                         use_PT_PL=use_PT_PL)

            hydro_simulations = dict((key, []) for key in hydro_names)
            for k, name in enumerate(hydro_names):
                for j, tau in enumerate(simulation_taus):
                    with open(('hydro_simulation_points/{}_simulation_points'
                               + '_n={}_tau={}.dat').
                              format(name, len(parameter_names), tau),
                              'r') as f_hydro_simulation_pts:
                        hydro_simulations[name].append(
                            [[float(entry)
                              for entry in line.split()]
                             for line in f_hydro_simulation_pts.readlines()
                             ])
            hydro_simulations = dict(
                (key, np.array(hydro_simulations[key]))
                for key in hydro_simulations)

            with open('pickle_files/emulator_testing_data.pkl', 'wb') as f:
                pickle.dump(hydro_simulations, f)

        print("From emulator validations: ",
              hydro_simulations['ce'].shape)
        print("Testing emulators")
        with open('full_outputs/emulator_test_n={}.txt', 'wb') as f:
            residuals_of_observables = {}
            for name in hydro_names:
                observable_residuals = []
                for i, tau in enumerate(simulation_taus):
                    for j, test_point in enumerate(self.test_points):
                        # Store and calculate observables from exact hydro
                        true_e = hydro_simulations[name][i, j, 1]
                        true_p1 = hydro_simulations[name][i, j, 2]
                        true_p2 = hydro_simulations[name][i, j, 3]

                        # calculate and store observables from emulator run
                        e = self.GP_emulators[name][i][0].predict(
                            np.array(test_point.
                                     reshape(1, -1))).reshape(1,)[0]
                        p1 = self.GP_emulators[name][i][1].predict(
                             np.array(test_point.
                                      reshape(1, -1))).reshape(1,)[0]
                        p2 = self.GP_emulators[name][i][2].predict(
                             np.array(test_point.
                                      reshape(1, -1))).reshape(1,)[0]
                        print(true_e, true_p1, true_p2)
                        print(e, p1, p2)
                        print(' ')

                        # calculate and store residuals
                        observable_residuals.append(
                            [e - true_e, p1 - true_p1, p2 - true_p2]
                        )

                # store all residuals for hydro `name`
                residuals_of_observables[name] = np.array(observable_residuals)
                print(np.array(observable_residuals).shape)

            # store residuals in DataFrame
            if use_PT_PL:
                p1_name = r'$R_{\mathcal P_T}$'
                p2_name = r'$R_{\mathcak P_L}$'
            else:
                p1_name = r'$R_\pi$'
                p2_name = r'$R_\Pi$'

            df = pd.DataFrame(columns=[r'$R_\mathcal{E}$',
                                       p1_name,
                                       p2_name,
                                       "hydro"])
            for name in hydro_names:
                df = pd.concat([df, pd.DataFrame(
                    {r'$R_\mathcal{E}$': residuals_of_observables[name][:, 0],
                     p1_name: residuals_of_observables[name][:, 1],
                     p2_name: residuals_of_observables[name][:, 2],
                     "hydro": [name] *
                        residuals_of_observables[name].shape[0]}
                )], ignore_index=True)

            # TODO: 1. Calculate means and standard deviations of all hydros
            #          and observables
            #       2. Output statistics to currently open data file
            #       3. Make plot of residuals and include means and mean
            #          percent error (of absolute value of residual)

        # output residuals to files
        with open('pickle_files/emulator_residuals_dataframe_n={}'.
                  format(len(parameter_names)), 'wb') as f:
            pickle.dump(df, f)

        print("Done")
