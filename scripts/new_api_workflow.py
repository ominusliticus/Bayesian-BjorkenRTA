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
# File: analysis_workflow.py
# Description: Provides convenience functions for modifying or accessing hydro
#              degrees of freedom

from hydro_bayesian_analysis import HydroBayesianAnalysis
from hydro_code_api import HydroCodeAPI
from hydro_emulation import HydroEmultion
from hydro_code_cmdln_options import HydroCodeCmdOptions
from hydro_code_utility import read_hydro_ouput
from hydro_code_utility import get_temp

from my_plotting import costumize_axis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.cm import plasma

from typing import Dict
from typing import Tuple
from typing import List
from pathlib import Path
from subprocess import run as cmd
from subprocess import CalledProcessError
import time

# Adjust plot fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def generate_hydro_data(
        output_path: Path,
        cmdln_options: HydroCodeCmdOptions,
) -> np.ndarray:
    '''
    Runs the C++ hydro code onces to and returns the output.
    Uses `cmdln_options` to customize what hydro simulation to run
    '''
    hca = HydroCodeAPI(output_path=output_path)
    hca.execute_hydro_code(cmdln_options=cmdln_options)
    return read_hydro_ouput(
        cmdln_options=cmdln_options,
        path_to_output=output_path)


def split_data_for_sequential_run(
    taus: np.ndarray,
    data: np.ndarray,
    error: np.ndarray,
) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]]:
    '''
    Takes and randomly splits hydro runs for sequential inferences
    '''
    rng = np.random.RandomState()

    entries = taus.shape[0]
    universal_set = np.arange(entries)
    training_indices = rng.choice(
        universal_set,
        size=entries // 2,
        replace=False
    )

    testing_indices = np.setdiff1d(
        universal_set, training_indices, assume_unique=True)
    taus_1 = taus[training_indices]
    taus_2 = taus[testing_indices]

    n_1 = np.argsort(taus_1)
    n_2 = np.argsort(taus_2)

    return (training_indices[n_1], testing_indices[n_2]), \
        (taus_1[n_1], taus_2[n_2]), \
        (data[training_indices][n_1], data[testing_indices][n_2]), \
        (error[training_indices][n_1], error[testing_indices][n_2])


def plot_exact_and_hydro_solns(
    output_path: Path,
    col_names: List[str],
    row_names: List[str],
    hydro_names: List[str],
    cmdln_options: HydroCodeCmdOptions,
    exact_soln: np.ndarray,
    exact_temp: np.ndarray,
    hydro_solns: np.ndarray,
):
    fig, ax = plt.subplots(
        nrows=len(hydro_names),
        ncols=3,
        figsize=(3 * 7, len(hydro_names) * 7)
    )
    fig.patch.set_facecolor('white')

    exact_tau_R = np.array([
        5 * cmdln_options['C'] / get_temp(
            energy_density=e,
            mass=cmdln_options['mass']
        )
        for e in exact_soln[:, 1]
    ])

    for j, col_name in enumerate(col_names):
        for i, hydro_name in enumerate(hydro_names):
            tau_R = np.array([
                5 * cmdln_options['C'] / get_temp(
                    energy_density=e,
                    mass=cmdln_options['mass']
                )
                for e in hydro_solns[i, ..., 1].reshape(-1,)
            ])
            ax[i, j].hist2d(
                hydro_solns[i, ..., 0].reshape(-1,) / tau_R,
                (hydro_solns[i, ..., j + 1].reshape(-1,)
                    / (hydro_solns[i, ..., 1]
                        + hydro_solns[i, ..., -1]).reshape(-1,)),
                bins=100,
                cmap=plasma,
                norm='log',
                alpha=0.5,
            )

            ax[i, j].plot(
                exact_soln[:, 0] / exact_tau_R,
                (exact_soln[i, j + 1]
                    / (exact_soln[i, 1]
                       + exact_soln[i, -1]).reshape(-1,)),
                color='black',
                lw=2,
            )
            costumize_axis(
                ax=ax[i, j],
                x_title=r'$\tau / \tau_R$',
                y_title=f'{col_name}'
            )
            ax[i, j].set_xlim(left=0)
            ax[i, j].text(0.9, 0.95, f'{hydro_name}', ha='left', va='top',
                          transform=ax[i, j].transAxes)
    plot_file = output_path / "plots/hydro_runs_for_posteriors.pdf"
    try:
        (cmd(['mkdir', '-p', str(plot_file.parent)])
            .check_returncode())
    except (CalledProcessError):
        print(f"Could not create dir {str(plot_file.parent)}")
    fig.tight_layout()
    fig.savefig(str(plot_file))


def main(
):
time_stamp = time.ctime(
