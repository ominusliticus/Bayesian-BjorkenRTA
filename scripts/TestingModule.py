#!/home/ominusliticus/anaconda3/bin/python3

from platform import uname
# import sys
import pickle
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from HydroBayesianAnalysis import HydroBayesianAnalysis

# My costumizations for plots
import matplotlib.ticker as tck
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)

# If on WSL set temporary directory
if 'microsoft' in uname().release:
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/'

def get_cmap(n: int, name: str='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def costumize_axis(ax: plt.Axes, x_title: str, y_title: str):
    ax.set_xlabel(x_title, fontsize=24)
    ax.set_ylabel(y_title, fontsize=24)
    ax.tick_params(axis='both', labelsize=18, top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='in', length=8)
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    return ax

def SampleObservables(error_level: float, exact_out: np.ndarray, pt_err: float, pl_err: float, p_err: float, b_fixed_detector_resolution: bool) -> np.ndarray:
    if b_fixed_detector_resolution:
        return np.full_like(exact_out, error_level)
    else:
        E, pt, pl = exact_out
        
        # energy density should not be normal distributed, but not other solution for now
        def YieldPositiveEnergyDensity() -> float:
            x = -np.inf
            while x < 0:
                x = np.random.normal(E, error_level * E)
            return x

        Ex = YieldPositiveEnergyDensity()

        ptx = np.random.normal(pt, np.fabs(pt_err))
        plx = np.random.normal(pl, np.fabs(pl_err))
        
        return np.array([Ex, ptx, plx]) 

def ConvertFromExactParametersToObservables(bayesian_inference: HydroBayesianAnalysis, params: np.ndarray):
    '''
    This function takes the MCMC chains from the MCMC run and converts them to the observables:
        - initial energy density
        - initial shear pressure
        - initial bulk pressure
    given the tau_0 and the relaxation time constant C

    Returns:
    --------
    tuple of (e0, pi0, Pi0)
    '''
    def GetExactResults() -> List:
            f_exact = open('../output/exact/MCMC_calculation_moments.dat','r')
            return f_exact.readlines()

    for i, name in enumerate(bayesian_inference.parameter_names):
        bayesian_inference.params[name] = params[i]
    
    bayesian_inference.params['tau_f'] = bayesian_inference.params['tau_0'] * (1 + 1 / 20.0)
    bayesian_inference.params['hydro_type'] = 4
    bayesian_inference.PrintParametersFile(bayesian_inference.params)
    bayesian_inference.RunHydroSimulation()
    t0, e0, pl0, pt0, p0 = [float(val) for val in GetExactResults()[0].split()]

    pi0 = (2 / 3) * (pt0 - pl0)
    Pi0 = (pl0 + 2 * pt0) / 3 - p0

    return params[0], t0, e0, pi0, Pi0

def PlotAnalyticPosteriors(Cs, E_exp, dE_exp, E_sim, P1_exp, dP1_exp, P1_sim, P2_exp, dP2_exp, P2_sim):
    '''
    We can write the analytic form of the posterior distributions and want to 
    to be able to compare these to those outputted by the MCMC walk
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    fig.patch.set_facecolor('white')

    size = Cs.size
    cmap = get_cmap(10, 'tab10')
    lines = [(0, (5, 10)), (0, (5, 5)), 'dashed', 'solid']
    for j, name in enumerate(['ce', 'dnmr', 'vah', 'mvah']):
        E_contrib = np.array([np.sum((E_exp[i] - E_sim[name][i]) ** 2 / dE_exp[i] ** 2) for i in range(size)])
        P1_contrib = np.array([np.sum((P1_exp[i] - P1_sim[name][i]) ** 2 / dP1_exp[i] ** 2) for i in range(size)])
        P2_contrib = np.array([np.sum((P2_exp[i] - P2_sim[name][i]) ** 2 / dP2_exp[i] ** 2) for i in range(size)])
        post = np.exp(- E_contrib - P1_contrib - P2_contrib) / (Cs[-1] - Cs[0])
        print(post)
        print(np.sum(post))
    
        ax.plot(Cs, post / np.sum(post), lw=2, ls=lines[j], color=cmap(j), label=name)
    ax.legend()
    costumize_axis(ax, r'$\mathcal C$', r'Posterior')
    fig.savefig("./plots/analytic_posteriors.pdf")

    return fig, ax

default_params =  {
    'tau_0':        0.1,
    'Lambda_0':     0.2 / 0.197,
    'xi_0':         -0.90, 
    'alpha_0':      0.655, #2 * pow(10, -3),
    'tau_f':        12.1,
    'mass':         1.015228426,
    'eta_s':        5 / (4 * np.pi),
    'hydro_type':   0
}

if __name__ == '__main__':
    # Flags for flow control of analysis:
    b_run_new_hydro = False          # If true, it tells HydroBayesAnalysis class to generate training points for GPs. 
    b_train_GP = False               # If true, HydroBayesAnalysis fits GPs to available training points
    b_read_mcmc = True             # If true, reads in last store MCMC chains
    b_calculate_observables = False # If true, reads in the observables (E, Pi, pi) calculated using the last MCMC chains
    
    print("Inside main function")

    # GP_parameter_names = ['eta_s','tau_0', 'Lambda_0', 'alpha_0', 'xi_0']
    # GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)], [0.05, 0.15], [0.0, 5.0], [0.0, 1.0], [-1.0, 10.0]])
    GP_parameter_names = ['eta_s']
    GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)]])
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
    bayesian_analysis_class_instance = HydroBayesianAnalysis(
        default_params=default_params,
        parameter_names=GP_parameter_names,
        parameter_ranges=GP_parameter_ranges,
        simulation_taus=simulation_taus,
        run_new_hydro=b_run_new_hydro,
        train_GP=b_train_GP
    )

    print("finished initialization of Bayesian Analysis class")
    # quit()
    if b_run_new_hydro and False:
        bayesian_analysis_class_instance.RunExactHydroForGPDesignPoints()

    exact_out = []
    # true_params = [5 / (4 * np.pi), 0.1, 0.2/.197, 2 * pow(10, -3), 0]
    true_params = [5 / (4 * np.pi)]
    bayesian_analysis_class_instance.params['tau_f'] = 12.1
    bayesian_analysis_class_instance.params['hydro_type'] = 4
    output = bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, true_params, store_whole_file=True)
    tau_start = 0.1
    delta_tau = tau_start / 20
    observ_indices = (simulation_taus - np.full_like(simulation_taus, tau_start)) / delta_tau

    exact_out = np.array([output[int(i)-1] for i in observ_indices])
    track_pt = np.array([output[int(i)-1,1] for i in observ_indices])
    track_pl = np.array([output[int(i)-1,2] for i in observ_indices])
    track_p = np.array([output[int(i)-1,3] for i in observ_indices])

    alpha_error = 0.10
    track_pt_err = alpha_error * track_pt
    track_pl_err = alpha_error * track_pl
    track_p_err = alpha_error * track_p
    exact_pseudo = np.zeros((simulation_taus.shape[0], 4))
    for i, tau in enumerate(simulation_taus):
        exact_pseudo[i, 0] = tau
        exact_pseudo[i, 1:4] = SampleObservables(alpha_error, exact_out[i, 1:4], track_pt_err[i], track_pl_err[i], track_p_err[i], False)

    pseudo_error = alpha_error * exact_out[:,1:4]
    
    bayesian_analysis_class_instance.RunMCMC(nsteps=2000, nburn=50, ntemps=10, exact_observables=exact_pseudo, exact_error=pseudo_error, read_from_file=b_read_mcmc)
    mcmc_chains = bayesian_analysis_class_instance.MCMC_chains
    evidences = bayesian_analysis_class_instance.evidence

    mcmc_observables = {}
    if b_calculate_observables:
        with open(f'pickle_files/mcmc_observables_n={len(GP_parameter_names)}.pkl', 'wb') as f:
            for name in bayesian_analysis_class_instance.hydro_names:
                mcmc_observables[name] = np.zeros((20 * len(GP_parameter_names), 200, 5))
                for i in range(mcmc_chains[name].shape[1]):
                    for j in range(mcmc_chains[name].shape[2]):
                        mcmc_observables[name][i, j, :] = ConvertFromExactParametersToObservables(bayesian_analysis_class_instance, mcmc_chains[name][0,i,j,:])
            pickle.dump(mcmc_observables, f)
    else:
        pass
        #with open(f'pickle_files/mcmc_observables_n={len(GP_parameter_names)}.pkl', 'rb') as f:
        #    mcmc_observables = pickle.load(f)

    cmap = get_cmap(10, 'tab10')
    if False:
        fig, ax = plt.subplots(nrows=1, ncols=len(GP_parameter_names), figsize=(len(GP_parameter_names) * 10,10))
        fig.patch.set_facecolor('white')
        
        if len(GP_parameter_names) == 1:
            axis_labels = [r'$\mathcal C$']
            limits = np.array([GP_parameter_ranges[0]])
            for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
                bins = np.linspace(*limits[0], 40, endpoint=True)
                ax.hist(mcmc_observables[name][:,:,0].flatten(), bins=bins, color=cmap(i), lw=2, histtype=u'step', label=name)
            costumize_axis(ax, axis_labels[0], r'Posterior')
            ax.legend(loc='upper right', fontsize=25)
            fig.tight_layout()
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists_observables.pdf')
        else:
            axis_labels = [r'$\mathcal C$', r'$\tau_0$', r'$\mathcal E_0$', r'$\pi_0$', r'$\Pi_0$']
            limits = np.array([GP_parameter_ranges[0], GP_parameter_ranges[1], [0, 10], [-3, 1], [-0.1, 0.01]])
            cmap = get_cmap(10, 'tab10')
            for n in range(len(axis_labels)):   
                for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
                    bins = np.linspace(*limits[n], 40, endpoint=True)
                    ax[n].hist(mcmc_observables[name][:,:,n].flatten(), bins=bins, color=cmap(i), lw=2, histtype=u'step', label=name)
                costumize_axis(ax[n], axis_labels[n], r'Posterior')
                if n == 2:
                     ax[n].legend(loc='upper right', fontsize=25)
                    #  ax[n].text(4, 4500, true_values_str, fontsize=20)
            fig.tight_layout()
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists_obseravble.pdf')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(GP_parameter_names), figsize=(len(GP_parameter_names) * 10,10))
        fig.patch.set_facecolor('white')
        
        if len(GP_parameter_names) == 1:
            axis_labels = [r'$\mathcal C$']
            limits = np.array([GP_parameter_ranges[0]])
            for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
                bins = np.linspace(*limits[0], 40, endpoint=True)
                sns.kdeplot(data=mcmc_chains[name][0,:,:].flatten(), ax=ax, color=cmap(i), lw=2, label=name)
                # ax.hist(mcmc_chains[name][0,:,:].flatten(), bins=bins, color=cmap(i), lw=2, histtype=u'step', label=name)
            costumize_axis(ax, axis_labels[0], r'Posterior')
            ax.legend(loc='upper right', fontsize=25)
            fig.tight_layout()
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists_parameters.pdf')
        else:
            axis_labels = [r'$\mathcal C$', r'$\tau_0$', r'$\mathcal E_0$', r'$\pi_0$', r'$\Pi_0$']
            limits = np.array([GP_parameter_ranges[0], GP_parameter_ranges[1], [0, 10], [-3, 1], [-0.1, 0.01]])
            cmap = get_cmap(10, 'tab10')
            for n in range(len(axis_labels)):   
                for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
                    bins = np.linspace(*limits[n], 40, endpoint=True)
                    ax[n].hist(mcmc_observables[name][0,:,:].flatten(), bins=bins, color=cmap(i), lw=2, histtype=u'step', label=name)
                costumize_axis(ax[n], axis_labels[n], r'Posterior')
                if n == 2:
                     ax[n].legend(loc='upper right', fontsize=25)
                    #  ax[n].text(4, 4500, true_values_str, fontsize=20)
            fig.tight_layout()
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists_parameters.pdf')

    map_outputs = {}
    map_values = {}
    bayesian_analysis_class_instance.params['tau_f'] = 12.1
    for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
        bayesian_analysis_class_instance.params['hydro_type'] = i
        n, b, p = plt.hist(mcmc_chains[name][0,:,:,:].flatten(), bins=1000)
        map_values[name] = [b[np.argmax(n)]] 
        map_outputs[name] = bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, map_values[name], store_whole_file=True)


    print(f'alpha_error = {alpha_error}')
    # calculating error bars from pseudo-data
    pseudo_e0 = np.random.normal(output[0,1], alpha_error * output[0,1])
    pseudo_e = exact_out[:,1]
    pseudo_e_err = alpha_error * pseudo_e

    pseudo_pt = exact_out[:,2]
    pseudo_pt_err = alpha_error * pseudo_pt

    pseudo_pl = exact_out[:,3]
    pseudo_pl_err = alpha_error * pseudo_pl

    pseudo_p = exact_out[:,4]
    pseudo_p_err = alpha_error * pseudo_p

    pseudo_pi = (2 / 3) * (pseudo_pt - pseudo_pl)
    pseudo_pi_err = (2 / 3) * np.sqrt(pseudo_pt_err ** 2 + pseudo_pl_err ** 2)

    pseudo_Pi = (2 * pseudo_pt + pseudo_pl) / 3 - pseudo_p
    pseudo_Pi_err = np.sqrt(4 * pseudo_pt_err ** 2 + pseudo_pl_err ** 2 + 9 * pseudo_p_err ** 2) / 3

    pseudo_h = pseudo_e + track_p
    pseudo_h_err = alpha_error * pseudo_h

    pseudo_e_e0 = pseudo_e / pseudo_e0
    pseudo_e_e0_err = alpha_error * pseudo_e_e0 * np.sqrt(2)

    pseudo_pi_bar = pseudo_pi / pseudo_h
    pseudo_pi_bar_err = pseudo_pi_bar * np.sqrt(pseudo_pi_err ** 2 / pseudo_pi ** 2 + pseudo_h_err ** 2 / pseudo_h ** 2)

    pseudo_Pi_bar = pseudo_Pi / pseudo_h
    pseudo_Pi_bar_err = pseudo_Pi_bar * np.sqrt(pseudo_Pi_err ** 2 / pseudo_Pi ** 2 + pseudo_h_err / pseudo_h ** 2)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30,20))
    fig.patch.set_facecolor('white')
    costumize_axis(ax[0,0], r'$\tau$ [fm]', r'$\mathcal E / \mathcal E_0$'); ax[0,0].set_yscale('log')
    costumize_axis(ax[0,1], r'$\tau$ [fm]', r'$\pi / (\mathcal E + \mathcal P)$')
    costumize_axis(ax[0,2], r'$\tau$ [fm]', r'$\Pi / (\mathcal E + \mathcal P)$')

    costumize_axis(ax[1,0], r'$\tau$ [fm]', r'$\mathcal E $ [fm$^{-4}$]'); ax[1,0].set_yscale('log')
    costumize_axis(ax[1,1], r'$\tau$ [fm]', r'$\pi $ [fm$^{-4}$]')
    costumize_axis(ax[1,2], r'$\tau$ [fm]', r'$\Pi $ [fm$^{-4}$]')

    ax[0,0].errorbar(simulation_taus, pseudo_e_e0, yerr=pseudo_e_e0_err, fmt='o', color='black', label='exact')
    ax[0,1].errorbar(simulation_taus, pseudo_pi_bar, yerr=pseudo_pi_bar_err, fmt='o', color='black') 
    ax[0,2].errorbar(simulation_taus, pseudo_Pi_bar, yerr=pseudo_Pi_bar_err, fmt='o', color='black')

    ax[1,0].errorbar(simulation_taus, pseudo_e, yerr=pseudo_e_err, fmt='o', color='black', label='exact')
    ax[1,1].errorbar(simulation_taus, pseudo_pi, yerr=pseudo_pi_err, fmt='o', color='black') 
    ax[1,2].errorbar(simulation_taus, pseudo_Pi, yerr=pseudo_Pi_err, fmt='o', color='black')


    for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
        print(f'map parameters for {name}: {map_values[name]}')
        output = map_outputs[name]
        t = output[:,0]
        e = output[:,1]
        p = output[:,4]
        pt = output[:,2]
        pl = output[:,3]
        pi = (2 / 3) * (pt - pl)
        Pi = (2 * pt + pl) / 3 - p
        pi_bar = pi / (e + p)
        Pi_bar = Pi / (e + p)
        ax[0,0].plot(t, e / e[0], lw=2, color=cmap(i), label=name)
        ax[0,1].plot(t, pi_bar, lw=2, color=cmap(i)); 
        ax[0,2].plot(t, Pi_bar, lw=2, color=cmap(i)); 

        ax[1,0].plot(t, e, lw=2, color=cmap(i), label=name)
        ax[1,1].plot(t, pi, lw=2, color=cmap(i))
        ax[1,2].plot(t, Pi, lw=2, color=cmap(i))
    ax[0,0].legend(loc='upper right', fontsize=20)
    fig.tight_layout()
    fig.savefig(f'plots/map_value_runs_n={len(GP_parameter_names)}.pdf')

    pts_analytic_post = 100
    hydro_names = ['ce', 'dnmr', 'vah', 'mvah']
    Cs = np.linspace(1 / (4 * np.pi), 10 / (4 * np.pi), pts_analytic_post, endpoint=True)
    for_analytic_hydro_output = dict((name, []) for name in hydro_names)
    bayesian_analysis_class_instance.params['tau_f'] = simulation_taus[-1]
    for i, name in enumerate(hydro_names):
        bayesian_analysis_class_instance.params['hydro_type'] = i
        output = np.array([[bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, [C], store_whole_file=True)[int(i)-1] for i in observ_indices] for C in Cs])
        for_analytic_hydro_output[name] = output
        
    for_analytic_hydro_output = dict((key, np.array(for_analytic_hydro_output[key])) for key in hydro_names)
    
    E_exp = np.array([pseudo_e for _ in range(pts_analytic_post)])
    dE_exp = np.array([pseudo_e_err for _ in range(pts_analytic_post)])

    PT_exp = np.array([pseudo_pt for _ in range(pts_analytic_post)])
    dPT_exp = np.array([pseudo_pt_err for _ in range(pts_analytic_post)])

    PL_exp = np.array([pseudo_pl for _ in range(pts_analytic_post)])
    dPL_exp = np.array([pseudo_pl_err for _ in range(pts_analytic_post)])

    pi_exp = np.array([(2/3) * (pseudo_pt - pseudo_pl) for _ in range(pts_analytic_post)])
    dpi_exp = np.array([(2/3) * np.sqrt(pseudo_pt_err ** 2 + pseudo_pl_err ** 2) for _ in range(pts_analytic_post)])

    Pi_exp = np.array([(2 * pseudo_pt + pseudo_pl) / 3 - pseudo_p for _ in range(pts_analytic_post)])
    dPi_exp = np.array([np.sqrt((4 * pseudo_pt_err ** 2 + pseudo_pl_err ** 2) / 9 + pseudo_p_err ** 2) for _ in range(pts_analytic_post)])

    E_sim = dict((key, for_analytic_hydro_output[key][:,:,1]) for key in hydro_names)
    P_sim = dict((key, for_analytic_hydro_output[key][:,:,4]) for key in hydro_names)
    PT_sim = dict((key, for_analytic_hydro_output[key][:,:,2]) for key in hydro_names)
    PL_sim = dict((key, for_analytic_hydro_output[key][:,:,3]) for key in hydro_names)
    pi_sim = dict((key, (2/3) * (PT_sim - PL_sim)) for key in hydro_names)
    Pi_sim = dict((key, (2 * PT_sim + PL_sim) / 3) for key in hydro_names)
    print(E_sim['ce'].shape, E_exp.shape)

    PlotAnalyticPosteriors(Cs=Cs,
                           E_exp=E_exp,
                           dE_exp=dE_exp,
                           E_sim=E_sim,
                           P1_exp=pi_exp,
                           dP1_exp=dpi_exp,
                           P1_sim=pi_sim,
                           P2_exp=Pi_exp,
                           dP2_exp=dPi_exp,
                           P2_sim=Pi_sim)

    
