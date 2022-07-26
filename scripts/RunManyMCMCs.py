#!/bin/python3
# My code
from HydroBayesianAnalysis import HydroBayesianAnalysis as HBA
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE
from my_plotting import costumize_axis, get_cmap, smooth_histogram

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

# For progress bars
from tqdm import tqdm

from multiprocessing import Process, Manager
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


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
    # ax.plot(points,
    #         gauss(points,
    #               loc=params[0],
    #               scale=params[1],
    #               norm=params[2]),
    #         lw=0.5,
    #         ls='solid',
    #         color='black')
    # fig.savefig(plot_name)


def normalize_for_C(xdata: np.ndarray,
                    ydata: np.ndarray,
                    scale: Optional[float] = None) -> np.ndarray:
    area = np.sum(np.diff(xdata)[0] * ydata)
    fit_params = fit_and_plot_posterior(xdata, ydata, None, None, None, None, None)
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
    true_params['hydro_type'] = 4
    output = code_api.ProcessHydro(params_dict=true_params,
                                   parameter_names=parameter_names,
                                   design_point=[true_params[key] for key in
                                                 parameter_names],
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


def RunManyMCMCRuns(exact_pseudo: np.ndarray,
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
    code_api = HCA(str(Path(output_dir + '/swap').absolute()))
    parameter_names = ['C']
    parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.linspace(5.1, 12.1, 8, endpoint=True)

    for i in tqdm(np.arange(start, n)):
        emulator_class = HE(hca=code_api,
                            params_dict=local_params,
                            parameter_names=parameter_names,
                            parameter_ranges=parameter_ranges,
                            simulation_taus=simulation_taus,
                            hydro_names=code_api.hydro_names,
                            use_existing_emulators=False,
                            use_PT_PL=True,
                            output_dir=output_dir,
                            samples_per_feature=points_per_feat)
        ba_class = HBA(default_params=local_params,
                       parameter_names=parameter_names,
                       parameter_ranges=parameter_ranges,
                       simulation_taus=simulation_taus)
        ba_class.RunMCMC(nsteps=400,
                         nburn=100,
                         ntemps=20,
                         exact_observables=exact_pseudo,
                         exact_error=pseudo_error,
                         GP_emulators=emulator_class.GP_emulators,
                         read_from_file=False)
        with open(output_dir + f'/mass_MCMC_run_{i}.pkl', 'wb') as f:
            pickle.dump(ba_class.MCMC_chains, f)


def RunVeryLargeMCMC(exact_pseudo: np.ndarray,
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
                        hydro_names=code_api.hydro_names,
                        use_existing_emulators=False,
                        use_PT_PL=True,
                        output_dir=output_dir,
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
    ba_class = HBA(default_params=local_params,
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

    # fig, ax = PlotAnalyticPosteriors(local_params=local_params,
    #                                  parameter_names=parameter_names,
    #                                  parameter_ranges=parameter_ranges,
    #                                  simulation_taus=simulation_taus,
    #                                  pseudo_data=exact_pseudo)
    ba_class.PlotPosteriors(output_dir=output_dir,
                            axis_names=[r'$\mathcal C$'])


def PlotAnalyticPosteriors(local_params: Dict[str, float],
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
    hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
    Cs = np.linspace(0.2,
                     0.5,
                     pts_analytic_post, endpoint=True)
    # TODO: Store calculated analytic result in pkl file...
    for_analytic_hydro_output = dict((name, []) for name in hydro_names)
    code_api = HCA(str(Path('./swap').absolute()))

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
        store_posteriors = dict((key, None)
                                for key in ['ce', 'dnmr', 'vah', 'mvah'])

        def calculate_gaussian(e: float,
                               pt: float,
                               pl: float,
                               de: float,
                               dpt: float,
                               dpl: float) -> float:
            val = np.exp(-e ** 2 / 2 / de ** 2)
            val *= np.exp(-pt ** 2 / 2 / dpt ** 2)
            val *= np.exp(-pl ** 2 / 2 / dpl ** 2)
            norm = np.sqrt((2 * np.pi) ** 3
                           * (de ** 2 * dpt ** 2 * dpl ** 2))
            return val / norm

        post = np.zeros_like(Cs)
        for j, name in enumerate(['ce', 'dnmr', 'vah', 'mvah']):
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

            store_posteriors[name] = post
            post_max = np.max(post)
            norms = np.array([5.6, 5.6, 4.78, 4.80])
            ax.plot(Cs,
                    post * norms[j] / post_max,
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
            print(np.sum(np.diff(Cs)[0] * line.get_ydata()))
    else:
        # Scan parameter space for hydro models
        manager = Manager()
        for_analytic_hydro_output = manager.dict()
        tau_start, delta_tau = 0.1, 0.005
        observ_indices = (simulation_taus
                          - np.full_like(simulation_taus,
                                         tau_start)) / delta_tau

        def for_multiprocessing(dic: Dict, key: str, itr: int):
            local_params['hydro_type'] = itr
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

        # for i, name in enumerate(hydro_names):
        #     local_params['hydro_type'] = i
        #     output = np.array(
        #         [[code_api.ProcessHydro(
        #               params_dict=local_params,
        #               parameter_names=parameter_names,
        #               design_point=[C],
        #               use_PT_PL=True)[int(i)-1]
        #           for i in observ_indices]
        #          for C in tqdm(Cs)])
        #     for_analytic_hydro_output[name] = output

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
            print(np.sum(np.diff(Cs) * line.get_ydata()))

    return fig, ax


def AverageManyRuns(output_dir: str,
                    runs: int) -> None:
    # TODO: generalize to a script that can read MAP from multi-dim
    #       parameter space
    out_dict = dict((key, np.zeros(runs))
                    for key in ['ce', 'dnmr', 'vah', 'mvah'])
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


def analyze_saved_runs(path_to_output: str,
                       number_of_runs: int,
                       posterior_fig: plt.Figure,
                       posterior_ax: plt.Axes) -> None:
    # range for hist binning
    hist_range = (0.2, 0.5)
    # data frames to be used by seaborn and draw posteriors
    dfs = pd.DataFrame(columns=[r'$\mathcal C$', 'hydro'])
    df_special = pd.DataFrame(columns=[r'$\mathcal C$', 'weight', 'hydro'])
    all_counts = dict((key, []) for key in ['ce', 'dnmr', 'vah', 'mvah'])
    x = np.random.randint(number_of_runs)
    for i in range(number_of_runs):
        with open(f'{path_to_output}/mass_MCMC_run_{i}.pkl', 'rb') as f:
            mcmc_chains = pickle.load(f)
            for key in mcmc_chains.keys():
                data = mcmc_chains[key][0].reshape(-1, 1)
                num_bins = 50 * int(np.sqrt(data.flatten().size))
                counts, bins = np.histogram(data.flatten(),
                                            bins=num_bins,
                                            range=hist_range,
                                            density=True)
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
                             # / np.sum(counts_special * np.diff(bins)),
                             'hydro': key})],
                        ignore_index=True)

    df_spread = pd.DataFrame(columns=[r'$\mathcal C$',
                                      r'$-\sigma$',
                                      r'$\mu$',
                                      r'$+\sigma$',
                                      'hydro'])

    bin_shift = np.diff(bins)[0]
    col_names = [r'$-\sigma$', r'$\mu$', r'$+\sigma$']
    for key in all_counts.keys():
        data = np.array(all_counts[key])
        high_low = np.quantile(data, [0.16, 0.50, 0.84], axis=0)
        print(high_low.shape)
        df = pd.DataFrame({
            r'$\mathcal C$': bins[:-1] + 0.5 * bin_shift,
            r'$-\sigma$': high_low[0],
            r'$\mu$': high_low[1],
            r'$+\sigma$': high_low[2],
            'hydro': key
        })
        df_spread = pd.concat([df_spread, df], ignore_index=True)

    num_lines_already = len(posterior_ax.get_lines())
    x_axis_col = r'$\mathcal C$'
    for name in col_names:
        # Doesn't work because we are just storing counts data
        # if name == r'$\mu':
        #     for hydro in all_counts.keys():
        #         sns.kdeplot(
        #             x=df_spread[df_spread['hydro'] == hydro][name],
        #             ax=posterior_ax,
        #             label=hydro)
        # else:
        #     for hydro in all_counts.keys():
        #         sns.kdeplot(
        #             x=df_spread[df_spread['hydro'] == hydro][name],
        #             ax=posterior_ax,
        #             linestyle='dashed'
        #         )
        if name == r'$\mu$':
            sns.kdeplot(
                data=df_spread,
                x=x_axis_col,
                weights=df_spread[name],
                hue='hydro',
                ax=posterior_ax,
                linewidth=1,
                common_norm=True,
                common_grid=True)
        else:
            sns.kdeplot(
                data=df_spread,
                x=x_axis_col,
                weights=df_spread[name],
                hue='hydro',
                linestyle='dashed',
                ax=posterior_ax,
                linewidth=1,
                common_norm=True,
                common_grid=True)

    lines_index = np.array([[0, 8],      # ce 1st std lines plotted
                            [1, 9],      # dmnr
                            [2, 10],     # vah
                            [3, 11]])    # mvah
    lines_index = lines_index + num_lines_already
    plotted_lines = posterior_ax.get_lines()

    fig_debug, ax_debug = plt.subplots(figsize=(7, 7))
    fig_debug.patch.set_facecolor('white')
    cmap = plt.get_cmap('tab10', 10)
    peaks = np.zeros(4)
    for k in range(num_lines_already):
        line = plotted_lines[k]
        x = line.get_xdata()
        y = line.get_ydata()
        peaks[k] = np.max(normalize_for_C(x, y))
        ax_debug.plot(x, normalize_for_C(x, y), lw=2, color=cmap(k))

    for k, pairs in enumerate(lines_index):
        x1 = plotted_lines[pairs[0]].get_xdata()
        y1 = plotted_lines[pairs[0]].get_ydata()
        x2 = plotted_lines[pairs[1]].get_xdata()
        y2 = plotted_lines[pairs[1]].get_ydata()

        posterior_ax.fill_between(
            x=x1,
            y1=y1,
            y2=y2,
            color=cmap(3-k),    # Lines seem to be fed like a queue
            alpha=0.4)

        print(np.sum(plotted_lines[pairs[0]].get_xdata() *
                     plotted_lines[pairs[0]].get_ydata()))
        print(np.sum(plotted_lines[pairs[1]].get_xdata() *
                     plotted_lines[pairs[1]].get_ydata()))

        x_mean = plotted_lines[pairs[0] + 4].get_xdata()
        y_mean = plotted_lines[pairs[0] + 4].get_ydata()
        fit_and_plot_posterior(
            xdata=x_mean,
            ydata=y_mean,
            points=x_mean,
            name=list(all_counts.keys())[3 - k],
            plot_name=path_to_output + '/plots/debug_posterior2.pdf',
            fig=posterior_fig,
            ax=posterior_ax)

        y1_new = normalize_for_C(x1, y1, peaks[3 - k])
        y2_new = normalize_for_C(x2, y2, peaks[3 - k])
        y_mean_new = normalize_for_C(x_mean, y_mean, peaks[3 - k])
        ax_debug.plot(x1, y1_new, lw=1, ls='dashed', color=cmap(3 - k))
        ax_debug.plot(x2, y2_new, lw=1, ls='dashed', color=cmap(3 - k))
        ax_debug.fill_between(x1, y1_new, y2_new, color=cmap(3 - k), alpha=0.5)
        ax_debug.plot(x_mean, y_mean_new, lw=1, ls='solid', color='black')

    posterior_ax.set_xlim(*hist_range)
    posterior_fig.tight_layout()
    posterior_fig.savefig(f'{path_to_output}/plots/upper-lower.pdf')

    ax_debug.set_xlim(*hist_range)
    costumize_axis(ax_debug, r'$\mathcal C$', 'Posterior')
    fig_debug.tight_layout()
    fig_debug.savefig(path_to_output + '/plots/same-norms-debug.pdf')

    g2 = sns.pairplot(data=dfs,
                      corner=True,
                      diag_kind='kde',
                      kind='hist',
                      hue='hydro')
    g2.map_lower(sns.kdeplot, levels=4)
    g2.tight_layout()
    g2.savefig(f'{path_to_output}/plots/full-posterior.pdf')
    del g2

    # compare one to all
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')
    sns.kdeplot(data=dfs,
                x=x_axis_col,
                hue='hydro',
                ax=ax)
    sns.kdeplot(data=df_special,
                x=x_axis_col,
                weights=df_special['weight'],
                hue='hydro',
                linestyle='dashed',
                alpha=0.5,
                ax=ax)
    ax.set_xlim(*hist_range)
    fig.tight_layout()
    fig.savefig(f'{path_to_output}/plots/compare-one-to-many.pdf')
    del fig, ax


if __name__ == "__main__":
    local_params = {
        'tau_0':        0.1,
        'Lambda_0':     0.5 / 0.197,
        'xi_0': -0.90,
        'alpha_0':      0.655,  # 2 * pow(10, -3),
        'tau_f':        12.1,
        'mass':         0.2 / 0.197,
        'C':            5 / (4 * np.pi),
        'hydro_type':   0
    }

    total_runs = 30
    # output_folder = 'very_large_mcmc_run_1'
    output_folder = 'Mass_run_5'

    # exact_pseudo, pseudo_error = SampleObservables(
    #     error_level=0.05,
    #     true_params=local_params,
    #     parameter_names=['C'],
    #     simulation_taus=np.linspace(5.1, 12.1, 8, endpoint=True))
    exact_pseudo = np.array(
        [[5.1,  0.75470255, 0.27463283, 0.1588341],
         [6.1,  0.6216813,  0.21947875, 0.14180029],
         [7.1,  0.51281854, 0.17400683, 0.11073481],
         [8.1,  0.39545993, 0.14676481, 0.10393984],
         [9.1,  0.40051311, 0.13026088, 0.09105533],
         [10.1, 0.30190729, 0.12180956, 0.07787765],
         [11.1, 0.30734799, 0.08858191, 0.07306867],
         [12.1, 0.25883392, 0.08667172, 0.06143159]]
    )
    pseudo_error = np.array(
        [[0.0383127,  0.013715,   0.00803337],
         [0.03082207, 0.01079027, 0.00672129],
         [0.02560243, 0.00879639, 0.00574629],
         [0.02177809, 0.00736298, 0.00499578],
         [0.01886813, 0.00629024, 0.00440209],
         [0.01658759, 0.00546173, 0.00392206],
         [0.0147574,  0.00480543, 0.00352685],
         [0.01325973, 0.00427459, 0.00319652]]
    )

    if True:
        # RunManyMCMCRuns(exact_pseudo=exact_pseudo,
        #                 pseudo_error=pseudo_error,
        #                 output_dir=f'./pickle_files/{output_folder}',
        #                 local_params=local_params,
        #                 points_per_feat=20,
        #                 n=total_runs,
        #                 start=0)

        # AverageManyRuns(output_dir=f'./pickle_files/{output_folder}',
        #                 runs=total_runs)

        fig, ax = PlotAnalyticPosteriors(
            local_params=local_params,
            parameter_names=['C'],
            parameter_ranges=np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]]),
            simulation_taus=np.linspace(5.1, 12.1, 8, endpoint=True),
            pseudo_data=exact_pseudo,
            pseudo_error=pseudo_error,
            path_to_output=f'./pickle_files/{output_folder}',
            use_existing_run=True)

        analyze_saved_runs(path_to_output=f'./pickle_files/{output_folder}',
                           number_of_runs=total_runs,
                           posterior_fig=fig,
                           posterior_ax=ax)
    else:
        RunVeryLargeMCMC(exact_pseudo=exact_pseudo,
                         pseudo_error=pseudo_error,
                         output_dir=f'./pickle_files/{output_folder}',
                         local_params=local_params,
                         points_per_feat=40,
                         number_steps=400)

    # analyze_saved_runs_hist(path_to_output=f'./pickle_files/{output_folder}',
    #                         number_of_runs=total_runs,
    #                         posterior_fig=fig_copy,
    #                         posterior_ax=ax_copy)
