#!/bin/python3
# My code
import code
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE

import numpy as np
from typing import Dict, List
from pathlib import Path
from subprocess import run

from tqdm import tqdm, trange

def RunManyEmulators(code_api: HCA,
                     local_params: Dict[str, float],
                     parameter_names: List[str],
                     parameter_ranges: List[List[float]],
                     simulation_taus: np.ndarray
                     ) -> None:
    for n in range(100):
        print(f'\r{n}')
        emulator_class = HE(hca=code_api,
                            params_dict=local_params,
                            parameter_names=parameter_names,
                            parameter_ranges=parameter_ranges,
                            simulation_taus=simulation_taus,
                            hydro_names=code_api.hydro_names,
                            use_existing_emulators=False,
                            use_PT_PL=True)
        
        emulator_class.TestEmulator(
            hca=code_api,
            params_dict=local_params,
            parameter_names=GP_parameter_names,
            parameter_ranges=GP_parameter_ranges,
            simulation_taus=simulation_taus,
            hydro_names=code_api.hydro_names,
            use_existing_emulators=False,
            use_PT_PL=True,
            output_statistics=True,
            plot_emulator_vs_test_points=True)
        
        run(['mv', 'plots/emulator_residuals_n=1.dat',
             f'plots/emulator_runs/emulator_residulas_n=1_{n}.dat'])
        run(['mv', 'plots/emulator_validation_plot_n=1.dat',
             f'plots/emulator_runs/emulator_validation_plot_n=1_{n}.dat'])
        run(['mv', 'pickle_files/emulator_testing_data_n=1.pkl',
             f'pickle_files/emulator_runs/emualor_testing_data_n=1_{n}.pkl'])
        run(['mv', 'pickle_files/emulator_residuals_dict_n=1.pkl',
             f'pickle_files/emulator_runs/emualor_resid_dict_n=1_{n}.pkl'])
        run(['mv', 'pickle_files/all_emulators_n=1.pkl',
             f'pickle_files/emulator_runs/all_emualors_n=1_{n}.pkl'])

if __name__ == '__main__':
    default_params = {
        'tau_0':        0.1,
        'Lambda_0':     0.5 / 0.197,
        'xi_0':         -0.90, 
        'alpha_0':      0.655, #2 * pow(10, -3),
        'tau_f':        12.1,
        'mass':         0.2 / 0.197,
        'C':            5 / (4 * np.pi),
        'hydro_type':   0
    }

    print("Running many emulators")
    GP_parameter_names = ['C']
    parameter_names_math = [r'$\mathcal C$']
    GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])

    code_api = HCA(str(Path('./swap').absolute()))
    RunManyEmulators(code_api=code_api,
                     local_params=default_params,
                     parameter_names=GP_parameter_names,
                     parameter_ranges=GP_parameter_ranges,
                     simulation_taus=simulation_taus)