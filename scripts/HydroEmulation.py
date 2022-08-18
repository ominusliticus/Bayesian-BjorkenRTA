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
# File: HydroEmulation.py
# Description: This file constructs the Gaussian emulators from hydro
#              simulations for efficient code evaluation

from platform import uname

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

# For plotting residuals
import matplotlib.pyplot as plt
from my_plotting import get_cmap, costumize_axis, autoscale_y

# for running hydro in parallel
from multiprocessing import Manager, Process

# For progress bars
from tqdm import tqdm

# for creating output directorie
from subprocess import run as cmd
from subprocess import CalledProcessError


class HydroEmulator:
    """
    This class trains the emulators for the various hydrodynamic models
    Per request, it also outputs diagnostic plots to evaluate the perfromance
    of the emulators

    Constructor parameters:
    ----------
    hca              - in instance of the HydroCodeAPI class
    params_dist      - dictionary with parameter names for hydro code
    parameter_names  - the named parameters which will be inferred upon
    parameter_ranges - ranges of value for which each parameter is defined
                     - assumes that the order is the same as for
                       parameter_names
    simulation_taus  - the times which to evaluate the emulators for
    hydro_name       - the hydro dynamic codes which to construct emulators for
                       possible values: 'ce', 'dnmr', 'vah', 'mvah'
    use_existing_emulators - bool to read in already trained emulators
    use_PL_PT        - flag for HydroCodeAPI that tells it whether to store
                     bulk and shear pressure, or longitudinal and transverse
                     pressure
    output_path      - path to output emulators to
    samples_per_feature - given n parameters in Bayesian inference, we will
                          generate samples_per_feature * n training points

    Returns
    ---------
    Dictionary of emulator for each hydro theory and simulation_tau
    """

    def __init__(self,
                 hca: HydroCodeAPI,
                 params_dict: Dict[str, float],
                 parameter_names: List[str],
                 parameter_ranges: np.ndarray,
                 simulation_taus: np.ndarray,
                 hydro_names: List[str],
                 use_existing_emulators: bool,
                 use_PT_PL: bool,
                 output_path: str,
                 samples_per_feature: int = 20
                 ):
        # creat directory for emualtors if it doesn't already exist
        try:
            cmd(['mkdir', '-p', output_path]).check_returncode()
        except (CalledProcessError):
            print(f'Failed to creat dir {output_path}')

        self.GP_emulators = dict((key, None) for key in hydro_names)
        if use_existing_emulators:
            # Load GP data from pickle files
            with open(
                    '{}/design_points_n={}.dat'.
                    format(output_path, len(parameter_names)),
                    'r'
                    ) as f:
                self.design_points = np.array(
                    [[float(entry) for entry in line.split()]
                     for line in f.readlines()])

            f_pickle_emulators = open(
                '{}/all_emulators_n={}.pkl'.
                format(output_path, len(parameter_names)),
                'rb')

            self.GP_emulators = pickle.load(f_pickle_emulators)
            f_pickle_emulators.close()
        else:
            try_until_no_nan = True
            while try_until_no_nan:
                # Run hydro code and generate scalers and GP pickle files
                # FIXME: I should sample testing points as well
                unit = lhs(n=len(parameter_names),
                           # add 10 points for testing data
                           samples=samples_per_feature * len(parameter_names),
                           criterion='maximin')
                self.design_points = parameter_ranges[:, 0] + unit * \
                    (parameter_ranges[:, 1] - parameter_ranges[:, 0])

                design_points = self.design_points
                self.test_points = np.linspace(parameter_ranges[:, 0],
                                               parameter_ranges[:, 1],
                                               10)
                # design_points = self.design_points[:-10]
                # self.test_points = self.design_points[-10:]
                with open('{}/design_points_n={}.dat'.
                          format(output_path, len(parameter_names)), 'w') as f:
                    for line in design_points.reshape(-1,
                                                      len(parameter_names)):
                        for entry in line:
                            f.write(f'{entry} ')
                        f.write('\n')
                with open('{}/testing_points_n={}.dat'.
                          format(output_path, len(parameter_names)), 'w') as f:
                    for line in self.test_points.reshape(-1,
                                                         len(parameter_names)):
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
                        with open(('{}/swap/{}_simulation_points'
                                   + '_n={}_tau={}.dat').
                                  format(output_path,
                                         name,
                                         len(parameter_names),
                                         tau),
                                  'r') as f_hydro_simulation_pts:
                            hydro_simulations[name].append(
                                [[float(entry)
                                  for entry in line.split()]
                                 for line in f_hydro_simulation_pts.readlines()
                                 ])
                hydro_simulations = dict(
                    (key, np.array(hydro_simulations[key]))
                    for key in hydro_simulations)

                nan_detected = np.array([0, 0, 0, 0], dtype=bool)
                for m, key in enumerate(hydro_names):
                    if np.any(np.isnan(hydro_simulations[key])):
                        nan_detected[m] = True
                        print('NaN detected, rerunning')
                    else:
                        nan_detected[m] = False
                try_until_no_nan = np.any(nan_detected)

            print("Fitting emulators")
            hydro_lists = np.array(
                [hydro_simulations[key] for key in hydro_names])

            obs = ['E', 'P1', 'P2']
            f_emulator_scores = open(
                f'{output_path}/emulator_scores_n={len(parameter_names)}.txt',
                'w')
            f_pickle_emulators = open(
                f'{output_path}/all_emulators_n={len(parameter_names)}.pkl',
                'wb')

            manager = Manager()
            self.GP_emulators = manager.dict()
            for name in hydro_names:
                self.GP_emulators[name] = []

            def train_hydro_emulator(global_emulators: Dict[str, List[gpr]],
                                     itr: int,
                                     name: str,
                                     hydro_lists: np.ndarray,
                                     parameter_ranges: np.ndarray,
                                     parameter_names: List[str],
                                     simulation_taus: np.ndarray,
                                     design_points: np.ndarray) -> None:
                all_emulators = []
                for j, tau in enumerate(tqdm(simulation_taus,
                                             desc=f'{name}: ',
                                             position=itr)):
                    local_emulators = []
                    f_emulator_scores.write(f'\tTraining GP for {name}\n')
                    for m in range(1, 4):
                        data = hydro_lists[itr, j, :, m].reshape(-1, 1)

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
                    all_emulators.append(local_emulators)
                global_emulators[name] = all_emulators

            # This seems like a programming pattern I can extract to anoterh
            # function
            if 'Darwin' in uname():
                for i, name in enumerate(self.GP_emulators.keys()):
                    train_hydro_emulator(global_emulators=self.GP_emulators,
                                         itr=i,
                                         name=name,
                                         hydro_lists=hydro_lists,
                                         parameter_ranges=parameter_ranges,
                                         parameter_names=parameter_names,
                                         simulation_taus=simulation_taus,
                                         desgin_points=design_points)
            else:
                jobs = [Process(target=train_hydro_emulator,
                                args=(self.GP_emulators,
                                      i,
                                      name,
                                      hydro_lists,
                                      parameter_ranges,
                                      parameter_names,
                                      simulation_taus,
                                      design_points))
                        for i, name in enumerate(hydro_names)]

                _ = [proc.start() for proc in jobs]
                _ = [proc.join() for proc in jobs]

            pickle.dump(self.GP_emulators, f_pickle_emulators)
            f_emulator_scores.close()
            f_pickle_emulators.close()

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
                     plot_emulator_vs_test_points: bool,
                     output_path: str) -> None:
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
            with open('{}/testing_points_n={}.dat'.
                      format(output_path, len(parameter_names)), 'r') as f:
                self.test_points = np.array(
                    [[float(entry)
                      for entry in line.split()]
                     for line in f.readlines()]
                )
            with open('{}/emulator_testing_data_n={}.pkl'.
                      format(output_path, len(parameter_names)), 'rb') as f:
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
                    with open(('{}/swap/{}_simulation_points'
                               + '_n={}_tau={}.dat').
                              format(output_path,
                                     name,
                                     len(parameter_names),
                                     tau),
                              'r') as f_hydro_simulation_pts:
                        hydro_simulations[name].append(
                            [[float(entry)
                              for entry in line.split()]
                             for line in f_hydro_simulation_pts.readlines()
                             ])
            hydro_simulations = dict(
                (key, np.array(hydro_simulations[key]))
                for key in hydro_simulations)

            with open('{}/emulator_testing_data_n={}.pkl'.
                      format(output_path, len(parameter_names)), 'wb') as f:
                pickle.dump(hydro_simulations, f)

        p1_name = r'$R_{\mathcal P_T}$' if use_PT_PL else r'$R_\pi$'
        p2_name = r'$R_{\mathcal P_L}$' if use_PT_PL else r'$R_\Pi$'

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
                            ax[k].plot(C, pred,
                                       lw=2, color=cmap(i), label=name)
                        else:
                            ax[k].plot(C, pred.reshape(-1,),
                                       lw=2, color=cmap(i))
                        ax[k].fill_between(C,
                                           pred[:] + err,
                                           pred[:] - err,
                                           color=cmap(i), alpha=.4)
            for k in range(3):
                autoscale_y(ax=ax[k], margin=0.1)
                costumize_axis(ax[k], r'$\mathcal C$', col_names[k])
            fig.legend(fontsize=18)
            fig.tight_layout()
            fig.savefig('{}/plots/emulator_validation_plot_n={}.pdf'.
                        format(output_path, len(parameter_names)))
            del fig, ax

        with open('{}/emulator_test_n={}.txt'.
                  format(output_path, len(parameter_names)), 'wb') as f:
            residuals_of_observables = {}
            print("Testing emulators")
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
            print("What is k here:", k)
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
            fig.savefig('{}/plots/emulator_residuals_n={}.pdf'.
                        format(output_path, len(parameter_names)))

        # output residuals to files
        with open('{}/emulator_residuals_dict_n={}.pkl'.
                  format(output_path, len(parameter_names)), 'wb') as f:
            pickle.dump(residuals_of_observables, f)

        print("Done")
