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
#
# Author: Kevin Ingles
# File: pymc_model_mixing.py
# Description: Doing local Bayesian Model Mixing using PyMC

import numpy as np

import pickle

from tqdm import tqdm

from pyDOE import lhs

from hydro_code_api import HydroCodeAPI
from hydro_emulation import HydroEmulator

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Iterable

from pathlib import Path

from multiprocessing import Manager
from multiprocessing import Process

from subprocess import run as cmd
from subprocess import CalledProcessError

from pytensor.tensor import TensorVariable
import pytensor.tensor as at

import pymc as pm

import arviz as az

import matplotlib.pyplot as plt
import my_plotting as mp

local_params = {
    'tau_0': 0.1,
    'e0': 12.4991,
    'pt0': 4.415,
    'pl0': 3.377,
    'tau_f': 12.1,
    'mass': 0.2 / 0.197,
    'C': 6 / (4 * np.pi),
    'hydro_type': 0
}


def generate_emulator_training_points(
    hydro_params_dict: Dict[str, float | int],
    hydro_parameter_names: List[str],
    hydro_parameter_ranges: np.ndarray,
    observation_times: np.ndarray,
) -> np.ndarray:
    hydro_names = ['ce', 'dnmr', 'mis', 'mvah']

    # instantiate HydroCodeAPI class
    output_dir = 'output_dir/emulator_design_points'
    code_api = HydroCodeAPI(str(Path(output_dir).absolute()))

    # latin hyper cube sample parameter space
    unit = lhs(
        n=len(hydro_inference_parameters),
        samples=10 * len(hydro_inference_parameters),
        criterion='maximin',
    )
    design_points = hydro_inference_parameters_ranges[:, 0] + \
        unit * np.diff(hydro_inference_parameters_ranges, axis=1)

    # run hydro simulations
    # this can be parallelized
    code_api.run_hydro(
        params_dict=hydro_params_dict,
        parameter_names=hydro_inference_parameters,
        design_points=design_points,
        simulation_taus=observation_times,
        hydro_names=hydro_names,
        use_PT_PL=False,
    )

    # extract the observables from the respective files
    # here is worth reminding myself that the HydroCodeAPI does the coalescing
    # of observables at observation times automatically
    hydro_simulations = dict((key, []) for key in hydro_names)
    for k, name in enumerate(hydro_names):
        for j, tau in enumerate(observation_times):
            with open(('{}/{}_simulation_points'
                       + '_n={}_tau={}.dat').
                      format(output_dir,
                             name,
                             len(hydro_inference_parameters),
                             tau),
                      'r') as f_hydro_simulation_pts:
                hydro_simulations[name].append(
                    [
                        [
                            float(entry)
                            for entry in line.split()
                        ]
                        for line in f_hydro_simulation_pts.readlines()
                     ]
                )
    hydro_simulations = dict(
        (key, np.array(hydro_simulations[key])[..., 0:4])
        for key in hydro_simulations
    )

    # return dictionary of observables
    return design_points, hydro_simulations


def generate_psuedo_data(
    error_level: float,
    true_params: Dict[str, Any],
    parameter_names: List[str],
    simulation_taus: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Runs RTA Boltzman solution and then smears it with a Gaussian with a std
    deviation proportional the central value (constant of proportionality given
    by the `error_level`)
    '''
    num_taus = simulation_taus.shape[0]
    code_api = HydroCodeAPI(str(Path('./swap').absolute()))

    # Generate experimental data
    true_params['hydro_type'] = 5
    output = code_api.process_hydro(params_dict=true_params,
                                    parameter_names=parameter_names,
                                    design_point=[true_params[key] for key in
                                                  parameter_names],
                                    use_PT_PL=True)
    tau_start = true_params['tau_0']
    delta_tau = tau_start / 20.0
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    e = np.array([output[int(i)-1, 1] for i in observ_indices])
    pt = np.array([output[int(i)-1, 2] for i in observ_indices])
    pl = np.array([output[int(i)-1, 3] for i in observ_indices])

    pt_err = error_level * pt
    pl_err = error_level * pl
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))

    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau

    ex = np.fabs(np.random.normal(e, np.fabs(e * error_level)))
    ptx = np.random.normal(pt, np.fabs(pt_err))
    plx = np.random.normal(pl, np.fabs(pl_err))

    return np.array(
        [
            (simulation_taus[i], ex[i], ptx[i], plx[i])
            for i in range(num_taus)
        ]
    ), error_level * np.array(
            [
                (e[i], pt[i], pl[i])
                for i in range(num_taus)
            ]
        )


def sequential_calibration(
    hydo_emulators: HydroEmulator,
    local_parameters: Dict[str, Any],
    data: np.ndarray,
    error_bar: np.ndarray
):
    '''
    Run calibration of individual models, then do model mixing
    '''
    # Use PyMC for individual calibrations and model mixing
    return NotImplemented


def simultaneous_calibration(
    hydro_names: List[str],
    hydro_inference_parameters: List[str],
    hydro_inference_parameters_ranges: np.ndarray,
    observation_times: np.ndarray,
    observation_data: np.ndarray,
    observation_error: np.ndarray,
    emulator_training_data: Dict[str, np.ndarray],
    emulator_design_points: np.ndarray
):
    '''
    Run calibration of model and model mixing simultaneously
    '''

    with pm.Model() as gp_emulators:
        # setup for gp emulator
        cov_func = pm.gp.cov.Matern32(
            input_dim=len(hydro_inference_parameters),
            ls=np.diff(hydro_inference_parameters_ranges, axis=1)
        )

        # dictionary to store all gps
        # some notes to remind me how PyMC gaussian processes seem to work
        # The process of training seems to take place when you marginalize over
        # the function prior (this gp.marginal_likelihood function call)
        # To predict, you can then use the stored gp and call the .predict
        # or .condtional methods and pass the new point
        # emulators = dict((key, []) for key in hydro_names)
        # for name in hydro_names:
        #     for i, training_data in enumerate(emulator_training_data[name]):
        #         observable_emulators = []
        #         for j, observable in enumerate(['e', 'pi', 'Pi']):
        #             observable_emulators.append(
        #                 pm.gp.Marginal(cov_func=cov_func)
        #             )
        #             observable_emulators[-1].marginal_likelihood(
        #                 name=f'{name}_{observable}_{i}',
        #                 X=emulator_design_points,
        #                 y=training_data[:, j + 1],
        #                 sigma=0,
        #             )
        #         emulators[name].append(observable_emulators)

    # To make predictions
    # x_new = np.linspace(
    #     start=hydro_inference_parameters_ranges[:, 0],
    #     stop=hydro_inference_parameters_ranges[:, 1],
    #     num=10,
    # ).reshape(-1, len(hydro_inference_parameters))
    # with gp_emulators:
    #     prediction = emulators['mvah'][0][0].predict(Xnew=x_new)
    # print(prediction)

    # I have verified for myself that the emulators look good, but it would
    # be nice to have a bit of code here that does the checking and plotting

    # Use PyMC for simulatneous calibrations (probably simpler)
    # return NotImplemented

    print(emulator_training_data['ce'].shape)
    print(observation_data.shape)


    with pm.Model() as inference_model:
        inference_vars = pm.Uniform(
            'hydro_inference_vars',
            lower=hydro_inference_parameters_ranges[:, 0],
            upper=hydro_inference_parameters_ranges[:, 1],
            shape=(len(hydro_inference_parameters), 1)
        )

        for i, tau in enumerate(observation_times):
            # comp_dists = [
            #     prod([
            #         # use costum dist here?
            #         pm.Normal.dist(
            #             # mu=emulators[name][i][j].predict(
            #             #     Xnew=sample,
            #             #     diag=True,
            #             # )[0],
            #             # sigma=np.sqrt(
            #             #     emulators[name][i][j].predict(
            #             #         Xnew=sample,
            #             #         # diag=True,
            #             #     )[1]
            #             #     +
            #             #     observation_error[i, j] ** 2
            #             # ),
            #             mu=pm.gp.Marginal(cov_func=cov_func).conditional(
            #                 name=f'{name}_{observable}_{i}',
            #                 Xnew=inference_vars,
            #                 given={
            #                     'X': emulator_design_points,
            #                     'y': emulator_training_data[name][i, :, j + 1],
            #                     'sigma': 0
            #                 },
            #                 shape=(len(hydro_inference_parameters), 1)
            #             ),
            #             sigma=observation_error[i, j],
            #         )
            #         for j, observable in enumerate(['e', 'pi', 'Pi'])
            #     ])
            #     for name in hydro_names
            # ]
            comp_dists = [
                pm.MvNormal.dist(
                    mu=[
                        pm.gp.Marginal(cov_func=cov_func).conditional(
                           name=f'{name}_{observable}_{i}',
                           Xnew=inference_vars,
                           given={
                               'X': emulator_design_points,
                               'y': emulator_training_data[name][i, :, j + 1],
                               'sigma': 0
                           },
                           shape=(len(hydro_inference_parameters),)
                        )
                        for j, observable in enumerate(['e', 'pi', 'Pi'])
                    ],
                    cov=np.diag(observation_error[i]),
                )
                for name in hydro_names
            ]

            alpha = pm.Lognormal(
                f'alpha_{i}',
                mu=0.0,
                sigma=1.0,
                shape=len(hydro_names)
            )
            weights = pm.Dirichlet(f'Dirichlet_{i}', a=alpha)
            pm.Mixture(
                f'mix_{i}',
                w=weights,
                comp_dists=comp_dists,
                observed=observation_data[i, 1:].reshape(-1,1),
            )

    with inference_model:
        pm.sample(1_000_000)


if __name__ == "__main__":
    true_params = local_params

    hydro_names = ['ce', 'dnmr', 'mis', 'mvah']
    hydro_inference_parameters = ['C']
    hydro_inference_parameters_ranges = np.array([[1, 10]]) / (4 * np.pi)
    simulation_taus = np.linspace(5.1, 12.1, 8)

    # Generate psuedo_data
    data, error_bar = generate_psuedo_data(
        error_level=0.05,
        parameter_names=hydro_inference_parameters,
        true_params=true_params,
        simulation_taus=simulation_taus,
    )

    print(data)

    # Generate training points to train emulator
    design_points, emulator_training_data = generate_emulator_training_points(
        hydro_params_dict=local_params,
        hydro_parameter_names=hydro_inference_parameters,
        hydro_parameter_ranges=hydro_inference_parameters_ranges,
        observation_times=simulation_taus,
    )

    print('\n\n\n\n')

    # sequential calibration

    # simultaneous calibration
    simultaneous_calibration(
        hydro_names=hydro_names,
        hydro_inference_parameters=hydro_inference_parameters,
        hydro_inference_parameters_ranges=hydro_inference_parameters_ranges,
        observation_times=simulation_taus,
        observation_data=data,
        observation_error=error_bar,
        emulator_training_data=emulator_training_data,
        emulator_design_points=design_points,
    )
