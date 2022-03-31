#!/bin/python3
# My code
from HydroBayesianAnalysis import HydroBayesianAnalysis as HBA
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE

import pickle
import numpy as np

def SampleObservables(error_level: float,
                      true_params: Dict[str, float],
                      simulation_taus: np.ndarray) -> np.ndarray:
    code_api = HCA(str(Path('./swap').absolute()))

    # Generate experimental data
    output = code_api.ProcessHydro(params_dict=true_params,
                                   parameter_names=GP_parameter_names,
                                   design_point=true_params,
                                   use_PT_PL=b_use_PT_PL)
    tau_start = 0.1
    delta_tau = tau_start / 20
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    exact_out = np.array([output[int(i)-1] for i in observ_indices])
    pt = np.array([output[int(i)-1, 1] for i in observ_indices])
    pl = np.array([output[int(i)-1, 2] for i in observ_indices])
    p = np.array([output[int(i)-1, 3] for i in observ_indices])

    error_level = 0.05
    pt_err = error_level * pt
    pl_err = error_level * pl
    p_err = error_level * p
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))

    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau

    E, pt, pl = exact_out

    # energy density should not be normal distributed, but not other
    # solution for now
    def YieldPositiveEnergyDensity() -> float:
        x = -np.inf
        while x < 0:
            x = np.random.normal(E, error_level * E)
        return x

    Ex = YieldPositiveEnergyDensity()

    ptx = np.random.normal(pt, np.fabs(pt_err))
    plx = np.random.normal(pl, np.fabs(pl_err))

    return np.array([Ex, ptx, plx]), error_level * np.array([E, pt, pl])

def RunManyMCMCRuns(output_dir: str,
                    n: int) -> None:
    '''
    Runs the entire analysis suite, including the emulator fiiting `n` times
    and saves MCMC chains in file indexed by the iteration number
    '''
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
    code_api = HCA(str(Path('./swap').absolute()))
    simulation_taus = np.linspace(5.1, 12.1, 8, endpoint=True)
    exact_psuedo, psuedo_error = SampleObservables(
        error_level=0.05,
        simulation_taus=simulation_taus)
    for i in np.arange(n):
        emulator_class = HE(hca=code_api,
                            params_dict=local_params,
                            parameter_names=GP_parameter_names,
                            parameter_ranges=GP_parameter_ranges,
                            simulation_taus=simulation_taus,
                            hydro_names=code_api.hydro_names,
                            use_existing_emulators=False,
                            use_PT_PL=False)
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
    RunManyMCMCRuns('./pickle_files', 500)
