#!/bin/python3
# My code
from HydroBayesianAnalysis import HydroBayesianAnalysis as HBA
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE
from my_plotting import costumize_axis, get_cmap

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from scipy.optimize import curve_fit

# For progress bars
from tqdm import tqdm

from multiprocessing import Process, Manager
from typing import Dict, List, Tuple, Optional
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
    print(f'm_p: {m_p}')
    print(f'beta_pi: {beta_pi}')
    print(f'beta_Pi: {beta_Pi}')

    # calculate eta and zeta
    eta = eta_s * m_s
    zeta = beta_Pi * eta / beta_pi
    print(f'eta: {eta}')
    print(f'zeta: {zeta}')

    # calculate P_L and P_T for navier-stokes initial conditions
    pt = m_p + ((2 / 3) * eta - zeta) / tau
    pl = m_p - ((4 / 3) * eta + zeta) / tau

    return pt, pl


def fit_and_plot_posterior(xdata: np.ndarray,
                           ydata: np.ndarray,
                           points: np.ndarray,
                           name: str,
                           plot_name: str,
                           fig: plt.Figure,
                           ax: plt.Axes) -> np.ndarray:
    def gauss(x: np.ndarray,
              loc: float,
              scale: float,
              norm: float) -> np.ndarray:
        return norm * np.exp(-(x - loc) ** 2 / (2 * scale ** 2))
    params, cov = curve_fit(gauss, xdata, ydata)
    print("fit params for analytic distribution {}: ({:.5e}, {:.5e})".
          format(name, params[0], params[1]))
    return params


def normalize_for_C(xdata: np.ndarray,
                    ydata: np.ndarray,
                    scale: Optional[float] = None) -> np.ndarray:
    area = np.sum(np.diff(xdata)[0] * ydata)
    fit_params = fit_and_plot_posterior(xdata, ydata,
                                        None, None, None, None, None)
    norm = np.sqrt(2 * np.pi * fit_params[1] ** 2)
    if not scale:
        return ydata / (area * norm)
    else:
        return ydata * scale / np.max(ydata)


def SampleObservables(error_level: float,
                      true_params: Dict[str, float],
                      parameter_names: List[str],
                      simulation_taus: np.ndarray) -> np.ndarray:
    num_taus = simulation_taus.shape[0]
    code_api = HCA(str(Path('./swap').absolute()))

    # Generate experimental data
    true_params['hydro_type'] = convert_hydro_name_to_int('exact')
    output = code_api.ProcessHydro(params_dict=true_params,
                                   parameter_names=parameter_names,
                                   design_point=[true_params[key] for key in
                                                 parameter_names],
                                   use_PT_PL=True)
    tau_start = 0.1
    delta_tau = tau_start / 20.0
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    E = np.array([output[int(i)-1, 1] for i in observ_indices])
    pt = np.array([output[int(i)-1, 2] for i in observ_indices])
    pl = np.array([output[int(i)-1, 3] for i in observ_indices])

    error_level = 0.05
    pt_err = error_level * pt
    pl_err = error_level * pl
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))

    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau

    Ex = np.fabs(np.random.normal(E, np.fabs(E * error_level)))
    ptx = np.random.normal(pt, np.fabs(pt_err))
    plx = np.random.normal(pl, np.fabs(pl_err))

    return np.array([(simulation_taus[i], Ex[i], ptx[i], plx[i])
                     for i in range(num_taus)]),\
        error_level * np.array([(E[i], pt[i], pl[i])
                                for i in range(num_taus)])


def RunManyMCMCRuns(hydro_names: List[str],
                    exact_pseudo: np.ndarray,
                    pseudo_error: np.ndarray,
                    output_dir: str,
                    local_params: Dict[str, float],
                    points_per_feat: int,
                    n: int,
                    start: int = 0) -> None:
    '''
    Runs the entire analysis suite, including the emulator fiiting `n` times
    and saves MCMC chains in file indexed by the iteration number
    '''
    code_api = HCA(str(Path(f'{output_dir}/swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.linspace(5.1, 12.1, 8, endpoint=True)

    for i in tqdm(np.arange(start, n), total=n):
        emulator_class = HE(hca=code_api,
                            params_dict=local_params,
                            parameter_names=parameter_names,
                            parameter_ranges=parameter_ranges,
                            simulation_taus=simulation_taus,
                            hydro_names=hydro_names,
                            use_existing_emulators=False,
                            use_PT_PL=True,
                            output_path=output_dir,
                            samples_per_feature=points_per_feat)
        ba_class = HBA(hydro_names=hydro_names,
                       default_params=local_params,
                       parameter_names=parameter_names,
                       parameter_ranges=parameter_ranges,
                       simulation_taus=simulation_taus)
        ba_class.RunMCMC(
            nsteps=400,
            nburn=100,
            ntemps=20,
            exact_observables=exact_pseudo,
            exact_error=pseudo_error,
            GP_emulators=emulator_class.GP_emulators,
            output_path=str(Path(f'{output_dir}/swap').absolute()),
            read_from_file=False)
        with open(output_dir + f'/mass_MCMC_run_{i}.pkl', 'wb') as f:
            pickle.dump(ba_class.MCMC_chains, f)


def RunVeryLargeMCMC(hydro_names: List[str],
                     exact_pseudo: np.ndarray,
                     pseudo_error: np.ndarray,
                     output_dir: str,
                     local_params: Dict[str, float],
                     points_per_feat: int,
                     number_steps: int) -> None:
    '''
    Runs the entire analysis suite, including the emulator fitting
    and saves MCMC chains and outputs plots
    '''
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.linspace(5.1, 12.1, 8, endpoint=True)

    emulator_class = HE(hca=code_api,
                        params_dict=local_params,
                        parameter_names=parameter_names,
                        parameter_ranges=parameter_ranges,
                        simulation_taus=simulation_taus,
                        hydro_names=hydro_names,
                        use_existing_emulators=False,
                        use_PT_PL=True,
                        output_path=output_dir,
                        samples_per_feature=points_per_feat)
    emulator_class.TestEmulator(
        hca=code_api,
        params_dict=local_params,
        parameter_names=parameter_names,
        parameter_ranges=parameter_ranges,
        simulation_taus=simulation_taus,
        hydro_names=code_api.hydro_names,
        use_existing_emulators=False,
        use_PT_PL=True,
        output_statistics=True,
        plot_emulator_vs_test_points=True,
        output_dir=output_dir)
    ba_class = HBA(hydro_names=hydro_names,
                   default_params=local_params,
                   parameter_names=parameter_names,
                   parameter_ranges=parameter_ranges,
                   simulation_taus=simulation_taus)
    ba_class.RunMCMC(nsteps=number_steps,
                     nburn=50,
                     ntemps=20,
                     exact_observables=exact_pseudo,
                     exact_error=pseudo_error,
                     GP_emulators=emulator_class.GP_emulators,
                     read_from_file=False)
    with open(output_dir + '/long_mcmc_run.pkl', 'wb') as f:
        pickle.dump(ba_class.MCMC_chains, f)

    ba_class.PlotPosteriors(output_dir=output_dir,
                            axis_names=[r'$\mathcal C$'])


def PlotAnalyticPosteriors(hydro_names: List[str],
                           local_params: Dict[str, float],
                           parameter_names: List[str],
                           parameter_ranges: Dict[str, np.ndarray],
                           simulation_taus: np.ndarray,
                           pseudo_data: np.ndarray,
                           pseudo_error: np.ndarray,
                           path_to_output: str,
                           use_existing_run: bool
                           ) -> Tuple[plt.Figure, plt.Axes]:
    '''
    We can write the analytic form of the posterior distributions and want to
    to be able to compare these to those outputted by the MCMC walk
    '''
    # Setup scan for inferred parameter
    pts_analytic_post = 100
    Cs = np.linspace(0.2,
                     0.5,
                     pts_analytic_post, endpoint=True)
    # TODO: Store calculated analytic result in pkl file...
    for_analytic_hydro_output = dict((name, []) for name in hydro_names)
    code_api = HCA(str(Path(f'{path_to_output}/swap').absolute()))

    # Generate experimental data
    pseudo_e = pseudo_data[:, 1]
    pseudo_e_err = pseudo_error[:, 0]

    pseudo_pt = pseudo_data[:, 2]
    pseudo_pt_err = pseudo_error[:, 1]

    pseudo_pl = pseudo_data[:, 3]
    pseudo_pl_err = pseudo_error[:, 2]

    def make_plot(for_analytic_hydro_output: np.ndarray
                  ) -> Tuple[plt.Figure, plt.Axes]:
        # prepare to make plots
        E_exp = np.array([pseudo_e for _ in range(pts_analytic_post)])
        dE_exp = np.array([pseudo_e_err for _ in range(pts_analytic_post)])

        PT_exp = np.array([pseudo_pt for _ in range(pts_analytic_post)])
        dPT_exp = np.array([pseudo_pt_err for _ in range(pts_analytic_post)])

        PL_exp = np.array([pseudo_pl for _ in range(pts_analytic_post)])
        dPL_exp = np.array([pseudo_pl_err for _ in range(pts_analytic_post)])

        E_sim = dict(
            (key, for_analytic_hydro_output[key][:, :, 1])
            for key in hydro_names)
        PT_sim = dict(
            (key, for_analytic_hydro_output[key][:, :, 2])
            for key in hydro_names)
        PL_sim = dict(
            (key, for_analytic_hydro_output[key][:, :, 3])
            for key in hydro_names)
        print(E_sim['ce'].shape, E_exp.shape)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        fig.patch.set_facecolor('white')

        size = Cs.size
        cmap = get_cmap(10, 'tab10')

        def calculate_gaussian(e: float,
                               pt: float,
                               pl: float,
                               de: float,
                               dpt: float,
                               dpl: float) -> float:
            val = np.exp(-e ** 2 / 2 / de ** 2)
            val *= np.exp(-pt ** 2 / 2 / dpt ** 2)
            val *= np.exp(-pl ** 2 / 2 / dpl ** 2)
            return val

        post = np.zeros_like(Cs)
        for j, name in enumerate(hydro_names):
            for i in range(size):
                temp = 1
                for k in range(simulation_taus.size):
                    temp *= calculate_gaussian(
                            e=(E_exp[i, k] - E_sim[name][i, k]),
                            pt=(PT_exp[i, k] - PT_sim[name][i, k]),
                            pl=(PL_exp[i, k] - PL_sim[name][i, k]),
                            de=dE_exp[i, k],
                            dpt=dPT_exp[i, k],
                            dpl=dPL_exp[i, k])
                post[i] = temp

            post_norm = np.sum(np.diff(Cs)[0] * post)
            ax.plot(Cs,
                    post / post_norm,
                    lw=2.5,
                    ls='solid',
                    color=cmap(j),
                    label=name)
        ax.legend(fontsize=20)
        costumize_axis(ax, r'$\mathcal C$', r'Posterior')
        fig.tight_layout()

        return fig, ax

    if use_existing_run:
        with open(path_to_output
                  + '/for_analytic_hydro_output.pkl', 'rb') as f:
            for_analytic_hydro_output = pickle.load(f)

        fig, ax = make_plot(for_analytic_hydro_output)
        lines = ax.get_lines()
        for k, line in enumerate(lines):
            fit_and_plot_posterior(
                xdata=line.get_xdata(),
                ydata=normalize_for_C(line.get_xdata(), line.get_ydata()),
                points=Cs,
                name=list(for_analytic_hydro_output.keys())[k],
                plot_name=path_to_output + '/plots/debug_posterior1.pdf',
                fig=fig,
                ax=ax)
    else:
        # Scan parameter space for hydro models
        manager = Manager()
        for_analytic_hydro_output = manager.dict()
        tau_start, delta_tau = 0.1, 0.1 / 20.0
        observ_indices = (simulation_taus
                          - np.full_like(simulation_taus,
                                         tau_start)) / delta_tau

        def for_multiprocessing(dic: Dict, key: str, itr: int):
            local_params['hydro_type'] = convert_hydro_name_to_int(key)
            output = np.array([[code_api.ProcessHydro(
                                    params_dict=local_params,
                                    parameter_names=parameter_names,
                                    design_point=[C],
                                    use_PT_PL=True)[int(i)-1]
                                for i in observ_indices]
                               for C in tqdm(Cs,
                                             desc=f'{key}: ',
                                             position=itr)])
            dic[key] = output

        jobs = [Process(target=for_multiprocessing,
                        args=(for_analytic_hydro_output, key, i))
                for i, key in enumerate(hydro_names)]
        _ = [proc.start() for proc in jobs]
        _ = [proc.join() for proc in jobs]

        for_analytic_hydro_output = dict(
            (key, np.array(for_analytic_hydro_output[key]))
            for key in hydro_names)
        with open(path_to_output
                  + '/for_analytic_hydro_output.pkl', 'wb') as f:
            pickle.dump(for_analytic_hydro_output, f)

        fig, ax = make_plot(for_analytic_hydro_output)
        lines = ax.get_lines()
        for k, line in enumerate(lines):
            fit_and_plot_posterior(
                xdata=line.get_xdata(),
                ydata=normalize_for_C(line.get_xdata(), line.get_ydata()),
                points=Cs,
                name=list(for_analytic_hydro_output.keys())[k],
                plot_name=path_to_output + '/plots/debug_posterior1.pdf',
                fig=fig,
                ax=ax)

    return fig, ax


def AverageManyRuns(hydro_names: List[str],
                    output_dir: str,
                    runs: int) -> None:
    # TODO: generalize to a script that can read MAP from multi-dim
    #       parameter space
    out_dict = dict((key, np.zeros(runs))
                    for key in hydro_names)
    for i in np.arange(runs):
        with open(output_dir + f'/mass_MCMC_run_{i}.pkl', 'rb') as f:
            mcmc_chains = pickle.load(f)
            for key in out_dict.keys():
                n, b = np.histogram(mcmc_chains[key][0, ...].flatten(),
                                    bins=1000,
                                    density=True)
                out_dict[key][i] = b[np.argmax(n)]

    avg = dict((key,
                np.sum(out_dict[key]) / runs)
               for key in out_dict.keys())
    std = dict((key,
                np.sqrt(np.sum(
                    np.power(out_dict[key] - avg[key], 2)) / (runs - 1)))
               for key in avg.keys())
    for key in avg.keys():
        print(f'{key}: {avg[key]} +/- {std[key]}')


def analyze_saved_runs(hydro_names: List[str],
                       path_to_output: str,
                       number_of_runs: int,
                       posterior_fig: plt.Figure,
                       posterior_ax: plt.Axes) -> None:
    # range for hist binning
    hist_range = (0.2, 0.5)
    # data frames to be used by seaborn and draw posteriors
    dfs = pd.DataFrame(columns=[r'$\mathcal C$', 'hydro'])
    df_special = pd.DataFrame(columns=[r'$\mathcal C$', 'weight', 'hydro'])
    all_counts = dict((key, []) for key in hydro_names)
    x = np.random.randint(number_of_runs)
    for i in range(number_of_runs):
        with open(f'{path_to_output}/mass_MCMC_run_{i}.pkl', 'rb') as f:
            mcmc_chains = pickle.load(f)
            for key in mcmc_chains.keys():
                data = mcmc_chains[key][0].reshape(-1, 1)
                num_bins = 1 * int(np.sqrt(data.flatten().size))
                counts, bins = np.histogram(data.flatten(),
                                            bins=num_bins,
                                            range=hist_range,
                                            density=False)
                all_counts[key].append(counts)
                df = pd.DataFrame({r'$\mathcal C$': data[:, 0]})
                df['hydro'] = key
                dfs = pd.concat([dfs, df], ignore_index=True)

                # need to create a histogram, and keep track of the bins
                # then plot the normalizes hist for the comparison
                if x == i:
                    counts_special, bins_special = np.histogram(
                        data,
                        bins=num_bins,
                        range=hist_range,
                        density=False)
                    df_special = pd.concat([
                        df_special,
                        pd.DataFrame(
                            {r'$\mathcal C$': bins_special[:-1],
                             'weight': counts_special,
                             'hydro': key})],
                        ignore_index=True)

    bin_shift = np.diff(bins)[0]
    norms = dict(
        (key,
         np.sum(
             bin_shift * np.quantile(
                 np.array(all_counts[key]), [0.5], axis=0)
         )) for key in all_counts.keys())

    cmap = get_cmap(10, 'tab10')
    x_bins = bins[:-1] + 0.5 * bin_shift
    for j, key in enumerate(norms.keys()):
        data = np.array(all_counts[key])
        high_low = np.quantile(data, [0.16, 0.50, 0.84], axis=0)
        for i in range(3):
            posterior_ax.scatter(x_bins,
                                 high_low[i] / norms[key],
                                 3.0,
                                 color=cmap(j)
                                 )
        posterior_ax.fill_between(
            x_bins,
            high_low[0] / norms[key],
            high_low[2] / norms[key],
            color=cmap(j),
            alpha=0.5
        )
        posterior_ax.plot(
            x_bins,
            high_low[1] / norms[key],
            lw=1,
            color=cmap(j)
        )

    posterior_ax.set_xlim(*hist_range)
    posterior_fig.tight_layout()
    posterior_fig.savefig(f'{path_to_output}/plots/upper-lower.pdf')


if __name__ == "__main__":
    # local_params = {
    #     'tau_0': 0.1,
    #     'e0': 12.4991,
    #     'pt0': 6.0977,
    #     'pl0': 0.0090,
    #     'tau_f': 12.1,
    #     'mass': 0.2 / 0.197,
    #     'C': 5 / (4 * np.pi),
    #     'hydro_type': 0
    # }

    # Navier-Stokes Initial Conditions
    e0 = 12.4991
    tau_0 = 5.0
    tau_f = 25.0
    eta_s = 5 / (4 * np.pi)
    mass = 0.2 / 0.197
    pt0, pl0 = get_navier_stokes_ic(e0, mass, eta_s, tau_0)
    print(f"pt0: {pt0}, pl0: {pl0}")
    local_params = {
        'tau_0': e0,
        'e0': e0,
        'pt0': pt0,
        'pl0': pl0,
        'tau_f': tau_f,
        'mass': mass,
        'C': eta_s,
        'hydro_type': 0
    }

    total_runs = 10
    # output_folder = 'very_large_mcmc_run_1'
    output_folder = 'Mass_run_NS'

    exact_pseudo, pseudo_error = SampleObservables(
        error_level=0.05,
        true_params=local_params,
        parameter_names=['C'],
        simulation_taus=np.linspace(5.1, 12.1, 8, endpoint=True))
    # exact_pseudo = np.array(
    #     [[5.1,  0.75470255, 0.27463283, 0.1588341],
    #      [6.1,  0.6216813,  0.21947875, 0.14180029],
    #      [7.1,  0.51281854, 0.17400683, 0.11073481],
    #      [8.1,  0.39545993, 0.14676481, 0.10393984],
    #      [9.1,  0.40051311, 0.13026088, 0.09105533],
    #      [10.1, 0.30190729, 0.12180956, 0.07787765],
    #      [11.1, 0.30734799, 0.08858191, 0.07306867],
    #      [12.1, 0.25883392, 0.08667172, 0.06143159]]
    # )
    # pseudo_error = np.array(
    #     [[0.0383127,  0.013715,   0.00803337],
    #      [0.03082207, 0.01079027, 0.00672129],
    #      [0.02560243, 0.00879639, 0.00574629],
    #      [0.02177809, 0.00736298, 0.00499578],
    #      [0.01886813, 0.00629024, 0.00440209],
    #      [0.01658759, 0.00546173, 0.00392206],
    #      [0.0147574,  0.00480543, 0.00352685],
    #      [0.01325973, 0.00427459, 0.00319652]]
    # )

    hydro_names = ['ce', 'dnmr', 'mis', 'mvah']

    if True:
        RunManyMCMCRuns(hydro_names=hydro_names,
                        exact_pseudo=exact_pseudo,
                        pseudo_error=pseudo_error,
                        output_dir=f'./pickle_files/{output_folder}',
                        local_params=local_params,
                        points_per_feat=20,
                        n=total_runs,
                        start=0)

        AverageManyRuns(hydro_names=hydro_names,
                        output_dir=f'./pickle_files/{output_folder}',
                        runs=total_runs)

        fig, ax = PlotAnalyticPosteriors(
            hydro_names=hydro_names,
            local_params=local_params,
            parameter_names=['C'],
            parameter_ranges=np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]]),
            simulation_taus=np.linspace(5.1, 12.1, 8, endpoint=True),
            pseudo_data=exact_pseudo,
            pseudo_error=pseudo_error,
            path_to_output=f'./pickle_files/{output_folder}',
            use_existing_run=False)

        analyze_saved_runs(hydro_names=hydro_names,
                           path_to_output=f'./pickle_files/{output_folder}',
                           number_of_runs=total_runs,
                           posterior_fig=fig,
                           posterior_ax=ax)
    else:
        RunVeryLargeMCMC(hydro_names=hydro_names,
                         exact_pseudo=exact_pseudo,
                         pseudo_error=pseudo_error,
                         output_dir=f'./pickle_files/{output_folder}',
                         local_params=local_params,
                         points_per_feat=40,
                         number_steps=400)

    # analyze_saved_runs_hist(path_to_output=f'./pickle_files/{output_folder}',
    #                         number_of_runs=total_runs,
    #                         posterior_fig=fig_copy,
    #                         posterior_ax=ax_copy)
