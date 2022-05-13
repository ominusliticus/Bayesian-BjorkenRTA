#!/bin/python3
# My code
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE

import numpy as np
from typing import Dict, List
from pathlib import Path
from subprocess import run

import PyPDF2
import pickle


def RunManyEmulators(code_api: HCA,
                     local_params: Dict[str, float],
                     parameter_names: List[str],
                     parameter_ranges: List[List[float]],
                     simulation_taus: np.ndarray,
                     move_files_to_long_term_storage: bool
                     ) -> None:
    for n in range(9, 100):
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

        if move_files_to_long_term_storage:
            run(['mv', 'plots/emulator_residuals_n=1.pdf',
                f'plots/emulator_runs/emulator_residulas_n=1_{n}.pdf'])
            run(['mv', 'plots/emulator_validation_plot_n=1.pdf',
                f'plots/emulator_runs/emulator_validation_plot_n=1_{n}.pdf'])
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

    hydro_names = [ 'ce', 'dnmr', 'vah', 'mvah' ]

    print("Running many emulators")
    GP_parameter_names = ['C']
    parameter_names_math = [r'$\mathcal C$']
    GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])

    code_api = HCA(str(Path('./swap').absolute()))
#     RunManyEmulators(code_api=code_api,
#                      local_params=default_params,
#                      parameter_names=GP_parameter_names,
#                      parameter_ranges=GP_parameter_ranges,
#                      simulation_taus=simulation_taus,
#                      move_files_to_long_term_storage=True)     

    
    # to do: 
    # 1. combine all residuals pdf plots to single pdf slideshow 
    #   Makes most sense to have a fixed y-axis 
    # 2. plot distributions of all residuals combined.
    # 3. plot average and max/min for hydro emulator evolutions (sus)
    residual_dict_file_string = 'emulator_resid_dict_n=1_{}.pkl'
    number_of_emulator_runs = 68

    # shape of residuals dictionary entry is 
    #  (n_observation_times, n_emulator_runs, n_observables)
    #  (8, 68, 3) for debuging code set
    dict_of_all_residuals = dict((key, None) for key in hydro_names)
    for i in range(number_of_emulator_runs):
        for key in hydro_names:
            with open(residual_dict_file_string.format(i), 'rb') as f:
                temp_dict = pickle.load(f)
                dict_of_all_residuals[key] = np.stack(
                                                 dict_of_all_residuals[key],
                                                 temp_dict[key])

    for i in range(dict_of_all_residuals[hydro_names[0]].shape[2]):
        fig, ax = 