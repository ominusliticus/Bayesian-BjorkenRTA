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
        self.GP_emulators = dict((key, None) for key in hydro_names)
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
                       samples=20 * len(parameter_names),
                       criterion='maximin')
            self.design_points = parameter_ranges[:, 0] + unit * \
                (parameter_ranges[:, 1] - parameter_ranges[:, 0])

            design_points = self.design_points
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
                        GPR.fit(design_points.reshape(-1,
                                                      len(parameter_names)),
                                data)

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
