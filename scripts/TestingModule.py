#!/home/ominusliticus/anaconda3/bin/python3

from platform import uname
# import sys
import pickle
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# facilitate path manipulation
from pathlib import Path

# My code
from HydroBayesianAnalysis import HydroBayesianAnalysis as HBA
from HydroCodeAPI import HydroCodeAPI as HCA
from HydroEmulation import HydroEmulator as HE

# My costumizations for plots
import matplotlib.ticker as tck
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

# If on WSL set temporary directory
if 'microsoft' in uname().release:
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/'


def get_cmap(n: int, name: str = 'hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)


def costumize_axis(ax: plt.Axes, x_title: str, y_title: str):
    ax.set_xlabel(x_title, fontsize=24)
    ax.set_ylabel(y_title, fontsize=24)
    ax.tick_params(axis='both', labelsize=18, top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='in', length=8)
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='minor',
                   direction='in', length=4, top=True, right=True)
    return ax


def SampleObservables(error_level: float,
                      exact_out: np.ndarray,
                      pt_err: float,
                      pl_err: float,
                      p_err: float,
                      b_fixed_error: bool) -> np.ndarray:
    if b_fixed_error:
        return np.full_like(exact_out, error_level)
    else:
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

        return np.array([Ex, ptx, plx]) 


def PlotAnalyticPosteriors(Cs,
                           E_exp, dE_exp, E_sim,
                           P1_exp, dP1_exp, P1_sim,
                           P2_exp, dP2_exp, P2_sim,
                           output_suffix):
    '''
    We can write the analytic form of the posterior distributions and want to
    to be able to compare these to those outputted by the MCMC walk
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    fig.patch.set_facecolor('white')

    size = Cs.size
    cmap = get_cmap(10, 'tab10')
    lines = [(0, (5, 10)), (0, (5, 5)), 'dashed', 'solid']
    for j, name in enumerate(['ce', 'dnmr', 'vah', 'mvah']):
        E_contrib = np.array(
            [np.sum((E_exp[i] - E_sim[name][i]) ** 2 / dE_exp[i] ** 2)
             for i in range(size)])

        P1_contrib = np.array(
            [np.sum((P1_exp[i] - P1_sim[name][i]) ** 2 / dP1_exp[i] ** 2)
             for i in range(size)])

        P2_contrib = np.array(
            [np.sum((P2_exp[i] - P2_sim[name][i]) ** 2 / dP2_exp[i] ** 2)
             for i in range(size)])

        post = np.exp(- E_contrib - P1_contrib - P2_contrib) / (Cs[1] - Cs[0])

        norm = np.sum(post) * (Cs[1] - Cs[0])
        ax.plot(Cs, post / norm, lw=2, ls=lines[j], color=cmap(j), label=name)
    ax.legend(fontsize=20)
    costumize_axis(ax, r'$\mathcal C$', r'Posterior')
    fig.tight_layout()
    fig.savefig(f"./plots/analytic_posteriors_{output_suffix}.pdf")

    return fig, ax

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

if __name__ == '__main__':
    # Flags for flow control of analysis:
    b_use_existing_emulators = False
    b_read_mcmc = False
    b_use_PT_PL = True

    print("Inside main function")

    # GP_parameter_names = ['C', 'tau_0', 'Lambda_0', 'alpha_0', 'xi_0']
    # parameter_names_math = [r'$\mathcal C$', r'$\tau_0$',
    #                         r'$\Lambda_0$', r'$\alpha_0$', r'$\xi_0$']
    # GP_parameter_ranges = np.array(
    #     [[1 / (4 * np.pi), 10 / (4 * np.pi)],
    #      [0.05, 0.15], [1.0, 5.0], [0.0, 1.0], [-1.0, 1.0]])
    GP_parameter_names = ['C']
    parameter_names_math = [r'$\mathcal C$']
    GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
    ba_class = HBA(
        default_params=default_params,
        parameter_names=GP_parameter_names,
        parameter_ranges=GP_parameter_ranges,
        simulation_taus=simulation_taus,
    )

    local_params = default_params

    exact_out = []
    # true_params = [5 / (4 * np.pi), 0.1, 0.5/.197, 0.655, -0.90]
    true_params = [5 / (4 * np.pi)]
    code_api = HCA(str(Path('./swap').absolute()))

    # Generate experimental data
    local_params['tau_f'] = 12.1
    local_params['hydro_type'] = 4
    output = code_api.ProcessHydro(params_dict=local_params,
                                   parameter_names=GP_parameter_names,
                                   design_point=true_params,
                                   use_PT_PL=b_use_PT_PL)
    tau_start = 0.1
    delta_tau = tau_start / 20
    observ_indices = (simulation_taus
                      - np.full_like(simulation_taus, tau_start)) / delta_tau

    exact_out = np.array([output[int(i)-1] for i in observ_indices])
    track_pt = np.array([output[int(i)-1, 1] for i in observ_indices])
    track_pl = np.array([output[int(i)-1, 2] for i in observ_indices])
    track_p = np.array([output[int(i)-1, 3] for i in observ_indices])

    error_level = 0.05
    track_pt_err = error_level * track_pt
    track_pl_err = error_level * track_pl
    track_p_err = error_level * track_p
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))
    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau
        exact_pseudo[i, 1:4] = SampleObservables(error_level=error_level,
                                                 exact_out=exact_out[i, 1:4],
                                                 pt_err=track_pt_err[i],
                                                 pl_err=track_pl_err[i],
                                                 p_err=track_p_err[i],
                                                 b_fixed_error=False)

    pseudo_error = error_level * exact_out[:, 1:4]

    # generate emulators
    emulator_class = HE(hca=code_api,
                        params_dict=local_params,
                        parameter_names=GP_parameter_names,
                        parameter_ranges=GP_parameter_ranges,
                        simulation_taus=simulation_taus,
                        hydro_names=code_api.hydro_names,
                        use_existing_emulators=b_use_existing_emulators,
                        use_PT_PL=b_use_PT_PL)
    emulator_class.TestEmulator(
        hca=code_api,
        params_dict=local_params,
        parameter_names=GP_parameter_names,
        parameter_ranges=GP_parameter_ranges,
        simulation_taus=simulation_taus,
        hydro_names=code_api.hydro_names,
        use_existing_emulators=b_use_existing_emulators,
        use_PT_PL=b_use_PT_PL)
    print("In between emulator function calls")
    ba_class.RunMCMC(nsteps=200,
                     nburn=50,
                     ntemps=10,
                     exact_observables=exact_pseudo,
                     exact_error=pseudo_error,
                     GP_emulators=emulator_class.GP_emulators,
                     read_from_file=b_read_mcmc)

    ba_class.PlotPosteriors(parameter_names_math)
    mcmc_chains = ba_class.MCMC_chains
    evidences = ba_class.evidence
    quit()

    if False:
        map_outputs = {}
        map_values = {}
        ba_class.params['tau_f'] = 12.1
        for i, name in enumerate(ba_class.hydro_names):
            ba_class.params['hydro_type'] = i
            n, b, p = plt.hist(mcmc_chains[name][0, :, :, :].flatten(),
                               bins=1000)
            map_values[name] = [b[np.argmax(n)]]
            map_outputs[name] = ba_class.ProcessHydro(GP_parameter_names,
                                                      map_values[name],
                                                      store_whole_file=True)

        print(f'error_level = {error_level}')
        # calculating error bars from pseudo-data
        pseudo_e0 = np.random.normal(output[0, 1], error_level * output[0, 1])
        pseudo_e = exact_out[:, 1]
        pseudo_e_err = error_level * pseudo_e

        pseudo_pt = exact_out[:, 2]
        pseudo_pt_err = error_level * pseudo_pt

        pseudo_pl = exact_out[:, 3]
        pseudo_pl_err = error_level * pseudo_pl

        pseudo_p = exact_out[:, 4]
        pseudo_p_err = error_level * pseudo_p

        pseudo_pi = (2 / 3) * (pseudo_pt - pseudo_pl)
        pseudo_pi_err = (2 / 3) * np.sqrt(pseudo_pt_err ** 2 +
                                          pseudo_pl_err ** 2)

        pseudo_Pi = (2 * pseudo_pt + pseudo_pl) / 3 - pseudo_p
        pseudo_Pi_err = np.sqrt(4 * pseudo_pt_err ** 2 +
                                pseudo_pl_err ** 2 +
                                9 * pseudo_p_err ** 2) / 3

        pseudo_h = pseudo_e + track_p
        pseudo_h_err = error_level * pseudo_h

        pseudo_e_e0 = pseudo_e / pseudo_e0
        pseudo_e_e0_err = error_level * pseudo_e_e0 * np.sqrt(2)

        pseudo_pi_bar = pseudo_pi / pseudo_h
        pseudo_pi_bar_err = pseudo_pi_bar * np.sqrt(pseudo_pi_err ** 2 /
                                                    pseudo_pi ** 2 +
                                                    pseudo_h_err ** 2 /
                                                    pseudo_h ** 2)

        pseudo_Pi_bar = pseudo_Pi / pseudo_h
        pseudo_Pi_bar_err = pseudo_Pi_bar * np.sqrt(pseudo_Pi_err ** 2 /
                                                    pseudo_Pi ** 2 +
                                                    pseudo_h_err /
                                                    pseudo_h ** 2)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
        fig.patch.set_facecolor('white')
        costumize_axis(ax[0, 0],
                       r'$\tau$ [fm]',
                       r'$\mathcal E / \mathcal E_0$')
        ax[0, 0].set_yscale('log')
        costumize_axis(ax[0, 1],
                       r'$\tau$ [fm]',
                       r'$\pi / (\mathcal E + \mathcal P)$')
        costumize_axis(ax[0, 2],
                       r'$\tau$ [fm]',
                       r'$\Pi / (\mathcal E + \mathcal P)$')

        costumize_axis(ax[1, 0],
                       r'$\tau$ [fm]',
                       r'$\mathcal E $ [fm$^{-4}$]')
        ax[1, 0].set_yscale('log')
        costumize_axis(ax[1, 1], r'$\tau$ [fm]', r'$\pi $ [fm$^{-4}$]')
        costumize_axis(ax[1, 2], r'$\tau$ [fm]', r'$\Pi $ [fm$^{-4}$]')

        ax[0, 0].errorbar(simulation_taus,
                          pseudo_e_e0,
                          yerr=pseudo_e_e0_err,
                          fmt='o',
                          color='black',
                          label='exact')
        ax[0, 1].errorbar(simulation_taus,
                          pseudo_pi_bar,
                          yerr=pseudo_pi_bar_err,
                          fmt='o',
                          color='black')
        ax[0, 2].errorbar(simulation_taus,
                          pseudo_Pi_bar,
                          yerr=pseudo_Pi_bar_err,
                          fmt='o',
                          color='black')

        ax[1, 0].errorbar(simulation_taus,
                          pseudo_e,
                          yerr=pseudo_e_err,
                          fmt='o',
                          color='black',
                          label='exact')
        ax[1, 1].errorbar(simulation_taus,
                          pseudo_pi,
                          yerr=pseudo_pi_err,
                          fmt='o',
                          color='black')
        ax[1, 2].errorbar(simulation_taus,
                          pseudo_Pi,
                          yerr=pseudo_Pi_err,
                          fmt='o',
                          color='black')

        cmap = get_cmap(10, 'tab10')
        for i, name in enumerate(ba_class.hydro_names):
            print(f'map parameters for {name}: {map_values[name]}')
            output = map_outputs[name]
            t = output[:, 0]
            e = output[:, 1]
            p = output[:, 4]
            pt = output[:, 2]
            pl = output[:, 3]
            pi = (2 / 3) * (pt - pl)
            Pi = (2 * pt + pl) / 3 - p
            pi_bar = pi / (e + p)
            Pi_bar = Pi / (e + p)
            ax[0, 0].plot(t, e / e[0], lw=2, color=cmap(i), label=name)
            ax[0, 1].plot(t, pi_bar, lw=2, color=cmap(i))
            ax[0, 2].plot(t, Pi_bar, lw=2, color=cmap(i))

            ax[1, 0].plot(t, e, lw=2, color=cmap(i), label=name)
            ax[1, 1].plot(t, pi, lw=2, color=cmap(i))
            ax[1, 2].plot(t, Pi, lw=2, color=cmap(i))
        ax[0, 0].legend(loc='upper right', fontsize=20)
        fig.tight_layout()
        fig.savefig(f'plots/map_value_runs_n={len(GP_parameter_names)}.pdf')

    # pts_analytic_post = 100
    # hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
    # Cs = np.linspace(1 / (4 * np.pi), 10 / (4 * np.pi), pts_analytic_post, endpoint=True)
    # for_analytic_hydro_output = dict((name, []) for name in hydro_names)
    # ba_class.params['tau_f'] = simulation_taus[-1]

    # Multiprocessing doesn't work because each call needs to write to a configuration file, and this can cause collision
    # for_analytic_hydro_output = Manager.dict()
    # def for_multiprocessing(dict: Dict, key: str, itr: int):
    #     ba_class.params['hydro_type'] = iter
    #     output = np.array([[ba_class.ProcessHydro(GP_parameter_names, [C], store_whole_file=True)[int(i)-1] for i in observ_indices] for C in Cs])
    #     dict[key] = output
    # 
    # jobs = [Process(for_multiprocessing, args=(for_analytic_hydro_output, key, i)) for i, key in enumerate(hydro_names)]
    # _ = [proc.start() for proc in jobs]
    # _ = [proc.join() for proc in jobs]

    # for i, name in enumerate(hydro_names):
    #     ba_class.params['hydro_type'] = i
    #     output = np.array([[ba_class.ProcessHydro(GP_parameter_names, [C], store_whole_file=True)[int(i)-1] for i in observ_indices] for C in Cs])
    #     for_analytic_hydro_output[name] = output
    #     
    # for_analytic_hydro_output = dict((key, np.array(for_analytic_hydro_output[key])) for key in hydro_names)
    # 
    # E_exp = np.array([pseudo_e for _ in range(pts_analytic_post)])
    # dE_exp = np.array([pseudo_e_err for _ in range(pts_analytic_post)])

    # PT_exp = np.array([pseudo_pt for _ in range(pts_analytic_post)])
    # dPT_exp = np.array([pseudo_pt_err for _ in range(pts_analytic_post)])

    # PL_exp = np.array([pseudo_pl for _ in range(pts_analytic_post)])
    # dPL_exp = np.array([pseudo_pl_err for _ in range(pts_analytic_post)])

    # pi_exp = np.array([(2/3) * (pseudo_pt - pseudo_pl) for _ in range(pts_analytic_post)])
    # dpi_exp = np.array([(2/3) * np.sqrt(pseudo_pt_err ** 2 + pseudo_pl_err ** 2) for _ in range(pts_analytic_post)])

    # Pi_exp = np.array([(2 * pseudo_pt + pseudo_pl) / 3 - pseudo_p for _ in range(pts_analytic_post)])
    # dPi_exp = np.array([np.sqrt((4 * pseudo_pt_err ** 2 + pseudo_pl_err ** 2) / 9 + pseudo_p_err ** 2) for _ in range(pts_analytic_post)])

    # E_sim = dict((key, for_analytic_hydro_output[key][:,:,1]) for key in hydro_names)
    # P_sim = dict((key, for_analytic_hydro_output[key][:,:,4]) for key in hydro_names)
    # PT_sim = dict((key, for_analytic_hydro_output[key][:,:,2]) for key in hydro_names)
    # PL_sim = dict((key, for_analytic_hydro_output[key][:,:,3]) for key in hydro_names)
    # pi_sim = dict((key, (2/3) * (PT_sim[key] - PL_sim[key])) for key in hydro_names)
    # Pi_sim = dict((key, (2 * PT_sim[key] + PL_sim[key]) / 3 - P_sim[key]) for key in hydro_names)
    # print(E_sim['ce'].shape, E_exp.shape)

    # PlotAnalyticPosteriors(Cs=Cs,
    #                        E_exp=E_exp,
    #                        dE_exp=dE_exp,
    #                        E_sim=E_sim,
    #                        P1_exp=pi_exp,
    #                        dP1_exp=dpi_exp,
    #                        P1_sim=pi_sim,
    #                        P2_exp=Pi_exp,
    #                        dP2_exp=dPi_exp,
    #                        P2_sim=Pi_sim,
    #                        output_suffix="Pipi")

    # PlotAnalyticPosteriors(Cs=Cs,
    #                        E_exp=E_exp,
    #                        dE_exp=dE_exp,
    #                        E_sim=E_sim,
    #                        P1_exp=PT_exp,
    #                        dP1_exp=dPT_exp,
    #                        P1_sim=PT_sim,
    #                        P2_exp=PL_exp,
    #                        dP2_exp=dPL_exp,
    #                        P2_sim=PL_sim,
    #                        output_suffix="PLPT")
