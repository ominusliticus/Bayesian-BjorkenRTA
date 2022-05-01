#
# Author: Kevin Ingles
# File: HydroEmulation.py
# Description: This file constructs the Gaussian emulators from hydro
#              simulations for efficient code evaluation

# For type identification
from enum import auto
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
import matplotlib.pyplot as plt
from my_plotting import get_cmap, costumize_axis, autoscale_y

# For progress bars
from tqdm import tqdm


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
                'pickle_files/all_emulators_n={}.pkl'.
                format(len(parameter_names)),
                'rb')

            self.GP_emulators = pickle.load(f_pickle_emulators)
            f_pickle_emulators.close()
        else:
            print("Running hydro")
            # Run hydro code and generate scalers and GP pickle files
            # FIXME: I should sample testing points as well
            unit = lhs(n=len(parameter_names),
                       # add 10 points for testing data
                       samples=20 * len(parameter_names), # + 10,
                       criterion='maximin')
            self.design_points = parameter_ranges[:, 0] + unit * \
                (parameter_ranges[:, 1] - parameter_ranges[:, 0])

            design_points = self.design_points
            self.test_points = np.linspace(parameter_ranges[:, 0],
                                           parameter_ranges[:, 1],
                                           10)
            # design_points = self.design_points[:-10]
            # self.test_points = self.design_points[-10:]
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

            print("Fitting emulators")
            hydro_lists = np.array(
                [hydro_simulations[key] for key in hydro_names])

            self.GP_emulators = dict((key, []) for key in hydro_names)

            obs = ['E', 'P1', 'P2']
            f_emulator_scores = open(
                f'full_outputs/emulator_scores_n={len(parameter_names)}.txt',
                'w')
            f_pickle_emulators = open(
                f'pickle_files/all_emulators_n={len(parameter_names)}.pkl',
                'wb')

            for i, name in enumerate(hydro_names):
                global_emulators = []
                for j, tau in enumerate(tqdm(simulation_taus,
                                             desc=f'{name}: ')):
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
                            print(f"ValueError encounter for {name}")
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
            f_pickle_emulators.close()

            print("Done")

    def TestEmulator(self,
                     hca: HydroCodeAPI,
                     params_dict: Dict[str, float],
                     parameter_names: List[str],
                     parameter_ranges: np.ndarray,
                     simulation_taus: np.ndarray,
                     hydro_names: List[str],
                     use_existing_emulators: bool,
                     use_PT_PL: bool,
                     output_statistics: bool,
                     plot_emulator_vs_test_points: bool) -> None:
        '''
        This function takes a given set of emulators and tests
        them for how accurately they run\n
        Parameters:
        ----------
        type annotations are specific enough

        Returns:
        ----------
        None
        '''
        if use_existing_emulators:
            with open('design_points/testing_points_n={}.dat'.
                      format(len(parameter_names)), 'r') as f:
                self.test_points = np.array(
                    [[float(entry)
                      for entry in line.split()]
                     for line in f.readlines()]
                )
            with open('pickle_files/emulator_testing_data_n={}.pkl'.
                      format(len(parameter_names)), 'rb') as f:
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

            with open('pickle_files/emulator_testing_data_n={}.pkl'.
                      format(len(parameter_names)), 'wb') as f:
                pickle.dump(hydro_simulations, f)

        if use_PT_PL:
            p1_name = r'$R_{\mathcal P_T}$'
            p2_name = r'$R_{\mathcal P_L}$'
        else:
            p1_name = r'$R_\pi$'
            p2_name = r'$R_\Pi$'

        col_names = [r'$R_\mathcal{E}$', p1_name, p2_name]
        # Make plot of emulators and test points
        if plot_emulator_vs_test_points:
            C = np.linspace(1 / (4 * np.pi), 10 / (4 * np.pi), 1000)
            feats = np.linspace(parameter_ranges[:, 0],
                                parameter_ranges[:, 1], 1000)
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(3 * 7, 7))
            fig.patch.set_facecolor('white')
            cmap = get_cmap(10, 'tab10')
            for i, name in enumerate(hydro_names):
                for j, tau in enumerate(simulation_taus):
                    for k in range(3):
                        pred, err = \
                            self.GP_emulators[name][j][k].predict(
                                feats, return_std=True)
                        if j == 0 and k == 0:
                            ax[k].plot(C, pred[:, 0], 
                                       lw=2, color=cmap(i), label=name)
                        else:
                            ax[k].plot(C, pred.reshape(-1,), 
                                       lw=2, color=cmap(i))
                        ax[k].fill_between(C,
                                           pred[:, 0] + err,
                                           pred[:, 0] - err,
                                           color=cmap(i), alpha=.4)
            for k in range(3):
                autoscale_y(ax=ax[k], margin=0.1)         
                costumize_axis(ax[k], r'$\mathcal C$', col_names[k])
            fig.legend(fontsize=18)
            fig.tight_layout()
            fig.savefig('plots/emulator_validation_plot_n={}.pdf'.
                        format(len(parameter_names)))
            del fig, ax

        with open('full_outputs/emulator_test_n={}.txt', 'wb') as f:
            residuals_of_observables = {}
            print(f"Testing emulators")
            for name in hydro_names:
                observable_residuals = []
                for i, tau in enumerate(tqdm(simulation_taus, 
                                        desc=f'{name}: ')):
                    local_list = []
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

                        # calculate and store residuals
                        local_list.append(
                            [(e - true_e) / true_e,
                             (p1 - true_p1) / true_p1,
                             (p2 - true_p2) / true_p2]
                        )
                    observable_residuals.append(local_list)

                # store all residuals for hydro `name`
                residuals_of_observables[name] = np.array(observable_residuals)

        if output_statistics:
            for name in hydro_names:
                print(f'Summary statistics for {name}')
                print('    energy density:')
                print('        mean abs % err:    {}'.
                      format(np.array([np.mean(np.abs(
                          residuals_of_observables[name][k, :, 0].
                          reshape(-1,)))])))
                print('        std abs % err:      {}'.
                      format(np.array([np.std(np.abs(
                          residuals_of_observables[name][k, :, 0].
                          reshape(-1,)))])))
                print(f'    {p1_name}:')
                print('        mean abs % err:    {}'.
                      format(np.array([np.mean(np.abs(
                          residuals_of_observables[name][k, :, 1].
                          reshape(-1,)))])))
                print('        std abs % err:      {}'.
                      format(np.array([np.std(np.abs(
                          residuals_of_observables[name][k, :, 1].
                          reshape(-1,)))])))
                print(f'    {p2_name}:')
                print('        mean abs % err:    {}'.
                      format(np.array([np.mean(np.abs(
                          residuals_of_observables[name][k, :, 2].
                          reshape(-1,)))])))
                print('        std abs % err:      {}'.
                      format(np.array([np.std(np.abs(
                          residuals_of_observables[name][k, :, 2].
                          reshape(-1,)))])))

            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(3 * 7, 7))
            fig.patch.set_facecolor('white')
            cmap = plt.get_cmap('tab10', 10)
            markers = ['o', '^', 's', 'p', 'H', 'x', '8', '*']
            for i in range(3):
                costumize_axis(ax[i], r'$\mathcal C$', col_names[i])
                for k, tau in enumerate(simulation_taus):
                    for j, name in enumerate(hydro_names):
                        if i == 0 and k == 0:
                            ax[i].plot(
                                self.test_points,
                                residuals_of_observables[name][k, :, i],
                                lw=2,
                                marker=markers[k],
                                color=cmap(j),
                                label=name)
                        else:
                            ax[i].plot(
                                self.test_points,
                                residuals_of_observables[name][k, :, i],
                                lw=1,
                                marker=markers[k],
                                color=cmap(j))
            autoscale_y(ax=ax[0], margin=0.1)
            autoscale_y(ax=ax[1], margin=0.1)
            autoscale_y(ax=ax[2], margin=0.1)
            fig.legend(fontsize=18)
            fig.tight_layout()
            fig.savefig('plots/emulator_residuals_n={}.pdf'.
                        format(len(parameter_names)))

        # output residuals to files
        with open('pickle_files/emulator_residuals_dict_n={}.pkl'.
                  format(len(parameter_names)), 'wb') as f:
            pickle.dump(residuals_of_observables, f)

        print("Done")
