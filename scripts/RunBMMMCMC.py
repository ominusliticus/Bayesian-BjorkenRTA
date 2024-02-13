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
from subprocess import run as cmd
from subprocess import CalledProcessError

from tqdm import tqdm

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

from matplotlib.cm import plasma


def split_data_for_sequential_run(
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState()

    entries = data.shape[0]
    training_indices_1 = rng.choice(
        np.arange(entries),
        size=entries // 2,
        replace=False
    )

    training_indices_2 = training_indices_1 - 1

    return data[training_indices_1], data[training_indices_2]


def convert_hydro_name_to_int(name: str) -> int:
    '''
    Returns integer corresponding to the hydro in C++ code.
    See documentation or ../src/main.cpp: int main() for options
    '''
    if name == 'ce':
        return 0
    if name == 'dnmr':
        return 1
    if name == 'mis':
        return 2
    if name == 'vah':
        return 3
    if name == 'mvah':
        return 4
    if name == 'exact':
        return 5


def run_hydro_from_posterior(
        mcmc_chains: Union[Dict[str, np.ndarray], np.ndarray],
        weights: np.ndarray,
        params_names: List[str],
        hydro_names: List[str],
        params_dict: Dict[str, Union[float, str]],
        ran_sequentially: bool,
        use_PL_PT: bool,
        output_dir: Path,
) -> None:
    code_api = HCA(str(Path('./swap').absolute()))
    params_dict['hydro_type'] = convert_hydro_name_to_int('exact')
    exact_output = code_api.process_hydro(
        params_dict=params_dict,
        parameter_names=params_names,
        design_point=[params_dict[key] for key in params_names],
        use_PL_PT=use_PL_PT
    )

    output_dict = dict((key, []) for key in hydro_names)
    if ran_sequentially:
        for name in hydro_names:
            params_dict['hydro_type'] = convert_hydro_name_to_int(name)
            for mcmc_step in tqdm(mcmc_chains[name]):
                output = code_api.process_hydro(
                    params_dict=params_dict,
                    parameter_names=parameter_names,
                    design_point=mcmc_step.reshape(-1,),
                    use_PL_PT=use_PL_PT
                )
                output_dict[name].append(output)
    else:
        for name in hydro_names:
            params_dict['hydro_type'] = convert_hydro_name_to_int(name)
            for mcmc_step in tqdm(mcmc_chains[:, len(hydro_names):]):
                output = code_api.process_hydro(
                    params_dict=params_dict,
                    parameter_names=parameter_names,
                    design_point=mcmc_step,
                    use_PL_PT=use_PL_PT
                )
                output_dict[name].append(output)

    output_array = np.array([
        output_dict[name] for name in hydro_names
    ])
    del output_dict

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(3 * 7, 3 * 7))
    fig.patch.set_facecolor('white')

    # fig2, ax2 = plt.subplots(nrows=1, ncols=3, figszie=(3 * 7, 7))
    # fig2.patch.set_facecolor('white')

    p1_name = r'${\mathcal P_T}$' if use_PL_PT else r'$\pi$'
    p2_name = r'${\mathcal P_L}$' if use_PL_PT else r'$\Pi$'

    col_names = [r'$\mathcal{E}$', p1_name, p2_name]
    for j, col_name in enumerate(col_names):
        for i, hydro_name in enumerate(hydro_names):
            ax[i, j].hist2d(
                output_array[i, ..., 0].reshape(-1,),
                output_array[i, ..., j + 1].reshape(-1,) * 0.197,
                bins=100,
                cmap=plasma,
                norm='log',
                alpha=0.5,
            )
            ax[i, j].plot(
                exact_output[:, 0],
                exact_output[:, j + 1] * 0.197,
                color='black',
                lw=2,
            )
            costumize_axis(
                ax=ax[i, j],
                x_title=r'$\tau$ [fm/c]',
                y_title=f'{col_name} [Gev/fm$^{-3}$]'
            )

        # ax2[j].hist2d(
            # output_dict[0, :, 0].reshape(-1),
            # (
                # weights.transpose() * output_dict[..., j + 1]
            # ).reshape(-1,),
            # bins=100,
            # cmap=plasma,
            # norm='log'
        # )
        # ax2[j].plot(
            # exact_output[:, 0],
            # exact_output[:, j + 1],
            # color='black',
            # lw=2,
        # )
        # costumize_axis(
            # ax=ax2[j],
            # x_title=r'$\tau$ [fm/c]',
            # y_title=f'{col_name} [gev/fm$^{-3}$]'
        # )

    fig.savefig(f'./pickle_files/{output_dir}/plots/hydro_runs_for_posteriors.pdf')
    # fig2.savefig(f'{output_dict}/plots/weight_average_of_hydro_runs_for_posteriors.pdf')



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


def RunVeryLargeMCMC(
        hydro_names: List[str],
        parameter_names: List[str],
        parameter_ranges: np.ndarray,
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
    ) -> Dict[str, np.ndarray]:
    '''
    Runs the entire analysis suite, including the emulator fitting
    and saves MCMC chains and outputs plots
    '''
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))

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
    mcmc_chains = ba_class.run_calibration(nsteps=number_steps,
                                           nburn=1000 * len(parameter_names),
                                           ntemps=20,
                                           true_observables=exact_pseudo,
                                           true_error=pseudo_error,
                                           GP_emulators=emulator_class.GP_emulators,
                                           read_from_file=read_mcmc_from_file,
                                           output_path=output_dir,
                                           run_parallel=True)
    with open(output_dir + '/long_mcmc_run.pkl', 'wb') as f:
        pickle.dump(ba_class.MCMC_chains, f)

    ba_class.plot_posteriors(output_dir=output_dir,
                             axis_names=[r'$\mathcal C$'])

    return mcmc_chains


def RunBMMMCMC(
    hydro_names: List[str],
    parameter_names: List[str],
    parameter_ranges: np.ndarray,
    simulation_taus: np.ndarray,
    exact_pseudo: np.ndarray,
    pseudo_error: np.ndarray,
    output_dir: str,
    local_params: Dict[str, float],
    points_per_feat: int,
    number_steps: int,
    fixed_values: Dict[str, float],
    use_existing_emulators: bool,
    read_mcmc_from_file: bool,
    use_PL_PT: bool,
    run_sequential: bool,
) -> None:
    '''
    Runs the entire analysis suite, including the emulator fitting
    and saves MCMC chains and outputs plots
    '''
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))

    emulator_class = HE(hca=code_api,
                        params_dict=local_params,
                        parameter_names=parameter_names,
                        parameter_ranges=parameter_ranges[len(hydro_names):]
                        .reshape(len(parameter_names), -1),
                        simulation_taus=simulation_taus,
                        hydro_names=hydro_names,
                        use_existing_emulators=use_existing_emulators \
                            and run_sequential,
                        use_PL_PT=use_PL_PT,
                        output_path=output_dir,
                        samples_per_feature=points_per_feat)

    if not use_existing_emulators:
        emulator_class.test_emulator(
            hca=code_api,
            params_dict=local_params,
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges[len(hydro_names):]
            .reshape(len(parameter_names), -1),
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

    bmm_mcmc_chains, weights = ba_class.run_mixing(
        nsteps=number_steps,
        nburn=1000 * len(parameter_ranges),
        ntemps=20,
        exact_observables=exact_pseudo,
        exact_error=pseudo_error,
        GP_emulators=emulator_class.GP_emulators,
        read_from_file=read_mcmc_from_file,
        do_calibration_simultaneous=(not run_sequential),
        fixed_evaluation_points_models=fixed_values,
        output_path=output_dir,
    )
    if not read_mcmc_from_file:
        with open(output_dir + '/bmm_mcmc_run.pkl', 'wb') as f:
            pickle.dump(ba_class.MCMC_chains, f)

    ba_class.plot_posteriors(output_dir=output_dir,
                             axis_names=[r'$\mathcal C$'])
    ba_class.plot_weights(output_dir=output_dir)

    return bmm_mcmc_chains, weights


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
    hydro_names = ['ce', 'dnmr', 'mvah']
    # hydro_names = ['ce', 'dnmr', 'mis', 'mvah']
    # hydro_names = ['mvah']

    # Weights parameters are not names explicitly
    # but we do explicitly includes the bounds for the wieghts
    parameter_names = ['C']
    parameter_ranges = np.array(
        [
            *[np.array([0, 10]) for _ in range(len(hydro_names))],
            [1 / (4 * np.pi), 10 / (4 * np.pi)]
        ]
    )

    # output_folder = 'bmm_runs_2/simultaneous_error=0.20'
    output_folder = 'bmm_runs_2/sequential_error=0.20'

    use_PL_PT = False
    generate_new_data = True
    use_existing_emulators = False
    read_mcmc_from_file = False
    run_sequential = True

    best_fits = [0.342, 0.40, 0.08, 0.235]
    simulation_taus = np.linspace(2.1, 4.1, 40, endpoint=True)

    data_file_path = Path(
        f'./pickle_files/{output_folder}/pseudo_data.pkl').absolute()
    try:
        (cmd(['mkdir', '-p', str(data_file_path.parent)])
            .check_returncode())
    except (CalledProcessError):
        print(f"Could not create dir {data_file_path}")
    if generate_new_data:
        with open(str(data_file_path), 'wb') as f:
            exact_pseudo, pseudo_error = SampleObservables(
                error_level=0.05,
                true_params=local_params,
                parameter_names=parameter_names,
                simulation_taus=simulation_taus,
                use_PL_PT=use_PL_PT,
            )
            pickle_output = (exact_pseudo, pseudo_error)
            pickle.dump(pickle_output, f)
    else:
        with open(str(data_file_path), 'rb') as f:
            pickle_input = pickle.load(f)
            exact_pseudo, pseudo_error = pickle_input

    # print(exact_pseudo)
    # print(pseudo_error)

    if run_sequential:
        simulation_taus_1, simulation_taus_2 = split_data_for_sequential_run(
            simulation_taus
        )
        exact_pseudo_1, exact_pseudo_2 = split_data_for_sequential_run(
            exact_pseudo
        )
        pseudo_error_1, pseudo_error_2 = split_data_for_sequential_run(
            pseudo_error
        )
        mcmc_chains = RunVeryLargeMCMC(
            hydro_names=hydro_names,
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges[len(hydro_names):],
            simulation_taus=simulation_taus_1,
          exact_pseudo=exact_pseudo_1,
            pseudo_error=pseudo_error_1,
            output_dir=f'./pickle_files/{output_folder}',
            local_params=local_params.copy(),
            points_per_feat=10,
            number_steps=20_000,
            use_existing_emulators=use_existing_emulators,
            read_mcmc_from_file=read_mcmc_from_file,
            use_PL_PT=use_PL_PT,
        )
        fixed_values = dict((name, np.mean(val[0]))
                            for name, val in mcmc_chains.items())
        # fixed_values = {
        #     'ce': [0.34138],
        #     'dnmr': [0.40024],
        #     'mvah': [0.3853]
        # }

    # TODO:
    #   - Add plotting for the posterior of the inference parameters when doing simultaneous calibration
    #   - Add plotting routine that plots the predictive posterior giving the weight average of the hydrodynamic theories and the exact solutions
    #   - Split large MCMC chains into smaller ones, se 10_000 steps at a time, and them combine them after everything has been run calculating the
    #       various quantities by looping over the separately stored runs
    bmm_mcmc_chains, weights  = RunBMMMCMC(
        hydro_names=hydro_names,
        simulation_taus=simulation_taus_2
        if run_sequential else simulation_taus,
        exact_pseudo=exact_pseudo_2
        if run_sequential else exact_pseudo,
        pseudo_error=pseudo_error_2
        if run_sequential else pseudo_error,
        parameter_names=parameter_names,
        parameter_ranges=parameter_ranges,
        output_dir=f'./pickle_files/{output_folder}',
        local_params=local_params.copy(),
        points_per_feat=10,
        number_steps=20_000,
        fixed_values=fixed_values if run_sequential else None,
        use_existing_emulators=use_existing_emulators,
        read_mcmc_from_file=read_mcmc_from_file,
        use_PL_PT=use_PL_PT,
        run_sequential=run_sequential,
    )

    if run_sequential:
        mcmc_chains = dict(
            (key, mcmc_chains[key][0].reshape(-1,))
            for key in hydro_names
        )
    bmm_mcmc_chains = bmm_mcmc_chains[0].reshape(
        -1,
        bmm_mcmc_chains.shape[-1]
    )
    weights = weights[0] # This still needs to be figured out, as weights has
                         # an extra dimension that keeps track of where the
                         # evaluation happened, ideally it'll promoted to a GP
    points_to_keep = 100
    run_hydro_from_posterior(
        mcmc_chains=dict(
            (
                key,
                mcmc_chains[key][
                    ::(mcmc_chains[key].shape[0] // points_to_keep)
                ]
            )
            for key in hydro_names
        ) if run_sequential else bmm_mcmc_chains[
            ::(bmm_mcmc_chains.shape[0] // points_to_keep)
        ],
        weights= 0,  # weights[::(weights.shape[1] // points_to_keep)],
        params_names=parameter_names,
        hydro_names=hydro_names,
        params_dict=local_params.copy(),
        ran_sequentially=run_sequential,
        use_PL_PT=use_PL_PT,
        output_dir=output_folder
    )
