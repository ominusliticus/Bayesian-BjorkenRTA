#!/bin/python3
# My code
from hydro_bayesian_analysis import HydroBayesianAnalysis as HBA
from hydro_code_api import HydroCodeAPI as HCA
from hydro_emulation import HydroEmulator as HE
from my_plotting import costumize_axis, get_cmap

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from scipy.optimize import curve_fit

from multiprocessing import Process, Manager
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def convert_hydro_name_to_int(name: str) -> int:
    '''
    Returns integer corresponding to the hydro in C++ code.
    See documentation or ../src/main.cpp: int main() for options
    '''
    match name:
        case 'ce':
            return 0
        case 'dnmr':
            return 1
        case 'mis':
            return 2
        case 'vah':
            return 3
        case 'mvah':
            return 4
        case 'exact':
            return 5


def get_navier_stokes_ic(
        energy_density: float, mass: float, eta_s: float, tau: float
) -> Tuple[float, float]:
    from scipy.special import kn
    from scipy.integrate import quad

    # calculate energy density given temperature and mass
    def e(temp, mass):
        z = mass / temp
        result = (z ** 2 * kn(2, z) / 2 + z ** 3 * kn(1, z) / 6)
        return 3 * temp ** 4 * result / np.pi ** 2

    # calculate equilibrium pressure given temperature and mass
    def p(temp, mass):
        z = mass / temp
        return z ** 2 * temp ** 4 * kn(2, z) / (2 * np.pi ** 2)

    # invert given energy density
    t_min = 0.001 / 0.197
    t_max = 2.0 / 0.197
    x_1 = t_min
    x_2 = t_max

    n = 0
    flag = 0
    copy = 0
    while flag != 1 and n <= 2000:
        mid = (x_1 + x_2) / 2.0
        e_mid = e(mid, mass)
        e_1 = e(x_1, mass)

        if np.abs(e_mid - energy_density) < 1e-6:
            break

        if (e_mid - energy_density) * (e_1 - e_mid) <= 0.0:
            x_2 = mid
        else:
            x_1 = mid

        n += 1
        if n == 1:
            copy = mid

        if n > 4:
            if np.abs(copy - mid) < 1e-6:
                flag = 1
            copy = mid

    # Function needed to calculate beta_pi
    def I_42_1(temp, mass):
        def k0n(z):
            return quad(
                lambda th: np.exp(- z * np.cosh(th)) / np.cosh(th),
                0, np.inf)[0]
        z = mass / temp
        result = (kn(5, z) - 7 * kn(3, z) + 2 * kn(1, z)) / 16 - k0n(z)
        return temp ** 5 * z ** 5 * result / (30 * np.pi ** 2)

    # define quantities necessary to calculate eta and zeta
    beta = 1 / mid
    m_e = energy_density
    m_p = p(mid, mass)
    m_s = (energy_density + m_p) / mid
    z = mass / mid
    cs2 = (m_e + m_p) / (3 * m_e + (3 + z ** 2) * m_p)
    beta_pi = beta * I_42_1(mid, mass)
    beta_Pi = (5 / 3) * beta_pi - (m_e + m_p) * cs2

    # calculate eta and zeta
    eta = eta_s * m_s
    zeta = beta_Pi * eta / beta_pi

    # calculate P_L and P_T for navier-stokes initial conditions
    pt = m_p + ((2 / 3) * eta - zeta) / tau
    pl = m_p - ((4 / 3) * eta + zeta) / tau

    return pt, pl


def SampleObservables(error_level: float,
                      true_params: Dict[str, float],
                      parameter_names: List[str],
                      simulation_taus: np.ndarray,
                      use_PL_PT: bool) -> np.ndarray:
    num_taus = simulation_taus.shape[0]
    code_api = HCA(str(Path('./swap').absolute()))

    # Generate experimental data
    true_params['hydro_type'] = convert_hydro_name_to_int('exact')
    output = code_api.process_hydro(params_dict=true_params,
                                    parameter_names=parameter_names,
                                    design_point=[true_params[key] for key in
                                                  parameter_names],
                                    use_PL_PT=use_PL_PT)
    tau_start = true_params['tau_0']
    delta_tau = tau_start / 20.0
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    # despite the poor variable choice name here, if `use_PL_PT` is False
    # all instances of (pt, pl) variables should be interpreted as (pi, Pi)
    E = np.array([output[int(i)-1, 1] for i in observ_indices])
    pt = np.array([output[int(i)-1, 2] for i in observ_indices])
    pl = np.array([output[int(i)-1, 3] for i in observ_indices])

    pt_err = error_level * pt
    pl_err = error_level * pl

    Ex = np.fabs(np.random.normal(E, np.fabs(E * error_level)))
    ptx = np.random.normal(pt, np.fabs(pt_err))
    plx = np.random.normal(pl, np.fabs(pl_err))

    return np.array([(simulation_taus[i], Ex[i], ptx[i], plx[i])
                     for i in range(num_taus)]), \
        error_level * np.array([(E[i], pt[i], pl[i])
                                for i in range(num_taus)])


def RunVeryLargeMCMC(hydro_names: List[str],
                     simulation_taus: np.ndarray,
                     exact_pseudo: np.ndarray,
                     pseudo_error: np.ndarray,
                     output_dir: str,
                     local_params: Dict[str, float],
                     points_per_feat: int,
                     number_steps: int,
                     use_existing_emulators: bool,
                     use_PL_PT: bool) -> None:
    '''
    Runs the entire analysis suite, including the emulator fitting
    and saves MCMC chains and outputs plots
    '''
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])

    emulator_class = HE(hca=code_api,
                        params_dict=local_params,
                        parameter_names=parameter_names,
                        parameter_ranges=parameter_ranges,
                        simulation_taus=simulation_taus,
                        hydro_names=hydro_names,
                        use_existing_emulators=use_existing_emulators,
                        use_PL_PT=use_PL_PT,
                        output_path=output_dir,
                        samples_per_feature=points_per_feat)
    emulator_class.test_emulator(
        hca=code_api,
        params_dict=local_params,
        parameter_names=parameter_names,
        parameter_ranges=parameter_ranges,
        simulation_taus=simulation_taus,
        hydro_names=hydro_names,
        use_existing_emulators=use_existing_emulators,
        use_PL_PT=use_PL_PT,
        output_statistics=True,
        plot_emulator_vs_test_points=True,
        output_path=output_dir)
    ba_class = HBA(hydro_names=hydro_names,
                   default_params=local_params,
                   parameter_names=parameter_names,
                   parameter_ranges=parameter_ranges,
                   simulation_taus=simulation_taus)
    ba_class.run_calibration(nsteps=number_steps,
                             nburn=50 * len(parameter_names),
                             ntemps=20,
                             exact_observables=exact_pseudo,
                             exact_error=pseudo_error,
                             GP_emulators=emulator_class.GP_emulators,
                             read_from_file=False,
                             output_path=output_dir)
    with open(output_dir + '/long_mcmc_run.pkl', 'wb') as f:
        pickle.dump(ba_class.MCMC_chains, f)

    ba_class.plot_posteriors(output_dir=output_dir,
                             axis_names=[r'$\mathcal C$'])


def RunBMMMCMC(
        hydro_names: List[str],
        simulation_taus: np.ndarray,
        exact_pseudo: np.ndarray,
        pseudo_error: np.ndarray,
        output_dir: str,
        local_params: Dict[str, float],
        points_per_feat: int,
        number_steps: int,
        use_existing_emulators: bool,
        read_mcmc_from_file: bool,
        use_PL_PT: bool
    ) -> None:
    '''
    Runs the entire analysis suite, including the emulator fitting
    and saves MCMC chains and outputs plots
    '''
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array(
        [
            *[np.array([0, 10]) for _ in range(len(hydro_names))],
            [1 / (4 * np.pi), 10 / (4 * np.pi)]
        ]
    )

    emulator_class = HE(hca=code_api,
                        params_dict=local_params,
                        parameter_names=parameter_names,
                        parameter_ranges=parameter_ranges[len(hydro_names):]
                                         .reshape(len(parameter_names),-1),
                        simulation_taus=simulation_taus,
                        hydro_names=hydro_names,
                        use_existing_emulators=use_existing_emulators,
                        use_PL_PT=use_PL_PT,
                        output_path=output_dir,
                        samples_per_feature=points_per_feat)
    
    if not use_existing_emulators:
        emulator_class.test_emulator(
            hca=code_api,
            params_dict=local_params,
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges[len(hydro_names):]
                            .reshape(len(parameter_names),-1),
            simulation_taus=simulation_taus,
            hydro_names=hydro_names,
            use_existing_emulators=use_existing_emulators,
            use_PL_PT=use_PL_PT,
            output_statistics=True,
            plot_emulator_vs_test_points=True,
            output_path=output_dir)
    ba_class = HBA(hydro_names=hydro_names,
                   default_params=local_params,
                   parameter_names=parameter_names,
                   parameter_ranges=parameter_ranges,
                   simulation_taus=simulation_taus,
                   do_bmm=True)
    
    ba_class.run_mixing(
        nsteps=number_steps,
        nburn=50 * len(parameter_names),
        ntemps=20,
        exact_observables=exact_pseudo,
        exact_error=pseudo_error,
        GP_emulators=emulator_class.GP_emulators,
        read_from_file=read_mcmc_from_file,
        do_calibration_simultaneous=False,
        fixed_evaluation_points_models={
                'ce': [0.34138],
                'dnmr': [0.40024],
                'mvah': [0.3853]
        },
        output_path=output_dir,
    )
    if not read_mcmc_from_file:
        with open(output_dir + '/bmm_mcmc_run.pkl', 'wb') as f:
            pickle.dump(ba_class.MCMC_chains, f)

    ba_class.plot_posteriors(output_dir=output_dir,
                             axis_names=[r'$\mathcal C$'])


if __name__ == "__main__":
    local_params = {
        'tau_0': 0.1,
        'e0': 12.4991,
        'pt0': 6.0977,
        'pl0': 0.0090,
        'tau_f': 12.1,
        'mass': 0.2 / 0.197,
        'C': 5 / (4 * np.pi),
        'hydro_type': 0
    }

    total_runs = 10

    # output_folder = 'very_large_mcmc_run_1'
    output_folder = 'bmm_runs/yes_bmm'
    
    use_PL_PT = False
    generate_new_data = False
    read_mcmc_from_file = True
    hydro_names = ['ce', 'dnmr', 'mvah']
    # hydro_names = ['ce', 'dnmr', 'mis', 'mvah']
    # hydro_names = ['mvah']

    best_fits = [0.342, 0.40, 0.08, 0.235]
    simulation_taus = np.linspace(2.1, 3.1, 10, endpoint=True)

    data_file_path = Path(f'./pickle_files/{output_folder}/pseudo_data.pkl').absolute()
    if generate_new_data:
        with open(str(data_file_path), 'wb') as f:
            exact_pseudo, pseudo_error = SampleObservables(
                error_level=0.0001,
                true_params=local_params,
                parameter_names=['C'],
                simulation_taus=simulation_taus,
                use_PL_PT=use_PL_PT,
            )
            pickle_output = (exact_pseudo, pseudo_error) 
            pickle.dump(pickle_output, f)
    else:
        with open(str(data_file_path), 'rb') as f:
            pickle_input = pickle.load(f)
            exact_pseudo, pseudo_error = pickle_input

    print(exact_pseudo)
    print(pseudo_error)

    RunBMMMCMC(hydro_names=hydro_names,
               simulation_taus=simulation_taus,
               exact_pseudo=exact_pseudo,
               pseudo_error=pseudo_error,
               output_dir=f'./pickle_files/{output_folder}',
               local_params=local_params,
               points_per_feat=10,
               number_steps=1_000,
               use_existing_emulators=True,
               read_mcmc_from_file=read_mcmc_from_file,
               use_PL_PT=use_PL_PT,)
