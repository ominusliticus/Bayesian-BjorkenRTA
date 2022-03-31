#!/bin/python3
# My code
from HydroBayesianAnalysis import HydroBayesianAnalysis as HBA
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE

import pickle
import numpy as np
from typing import Dict, List
from pathlib import Path

def SampleObservables(error_level: float,
                      true_params: Dict[str, float],
                      parameter_names: List[str],
                      simulation_taus: np.ndarray) -> np.ndarray:
    code_api = HCA(str(Path('./swap').absolute()))

    # Generate experimental data
    true_params['hydro_type'] = 4
    output = code_api.ProcessHydro(params_dict=true_params,
                                   parameter_names=parameter_names,
                                   design_point=list(true_params.values()),
                                   use_PT_PL=True)
    tau_start = 0.1
    delta_tau = tau_start / 20
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    exact_out = np.array([output[int(i)-1] for i in observ_indices])
    print(exact_out.shape)
    E = np.array([output[int(i)-1, 1] for i in observ_indices])
    pt = np.array([output[int(i)-1, 2] for i in observ_indices])
    pl = np.array([output[int(i)-1, 3] for i in observ_indices])
    p = np.array([output[int(i)-1, 4] for i in observ_indices])

    error_level = 0.05
    pt_err = error_level * pt
    pl_err = error_level * pl
    p_err = error_level * p
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))

    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau

    Ex = np.fabs(np.random.normal(E, np.fabs(E * error_level)))
    ptx = np.random.normal(pt, np.fabs(pt_err))
    plx = np.random.normal(pl, np.fabs(pl_err))

    return np.array([Ex, ptx, plx]), error_level * np.array([E, pt, pl])

def RunManyMCMCRuns(output_dir: str,
                    local_params: Dict[str, float],
                    n: int) -> None:
    '''
    Runs the entire analysis suite, including the emulator fiiting `n` times
    and saves MCMC chains in file indexed by the iteration number
    '''
    code_api = HCA(str(Path('./swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.linspace(5.1, 12.1, 8, endpoint=True)
    exact_psuedo, psuedo_error = SampleObservables(
        error_level=0.05,
        true_params=local_params,
        parameter_names=parameter_names,
        simulation_taus=simulation_taus)
    for i in np.arange(n):
        emulator_class = HE(hca=code_api,
                            params_dict=local_params,
                            parameter_names=parameter_names,
                            parameter_ranges=parameter_ranges,
                            simulation_taus=simulation_taus,
                            hydro_names=code_api.hydro_names,
                            use_existing_emulators=False,
                            use_PT_PL=True)
        ba_class.RunMCMC(nsteps=200,
                         nburn=50,
                         ntemps=10,
                         exact_observables=exact_pseudo,
                         exact_error=pseudo_error,
                         GP_emulators=emulator_class.GP_emulators,
                         read_from_file=False)
        with open(output_dir + f'/mass_MCMC_run_{i}.pkl', 'wb') as f:
            pickle.dump(ba_class.MCMC_chains, f)

if __name__ == "__main__":
    local_params =  {
        'tau_0':        0.1,
        'Lambda_0':     0.2 / 0.197,
        'xi_0':         -0.90, 
        'alpha_0':      0.655, #2 * pow(10, -3),
        'tau_f':        12.1,
        'mass':         1.015228426,
        'C':            5 / (4 * np.pi),
        'hydro_type':   0
    }

    RunManyMCMCRuns(output_dir='./pickle_files',
                    local_params=local_params,
                    n=3)
