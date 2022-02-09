#!/home/ominusliticus/anaconda3/bin/python3

from platform import uname

from numpy.lib import stride_tricks

# If on WSL set temporary directory
if 'microsoft' in uname().release:
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/'

import sys

import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.stats import poisson

from HydroBayesianAnalysis import *

# My costumizations for plots
import matplotlib.ticker as tck
import matplotlib.font_manager
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)

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

def SampleObservables(error_level: float, exact_out: np.ndarray, b_fixed_detector_resolution: bool) -> np.ndarray:
    if b_fixed_detector_resolution:
        return np.full_like(exact_out, error_level)
    else:
        E, pi, Pi = exact_out
        
        # energy density should not be normal distributed, but not other solution for now
        def YieldPositiveEnergyDensity() -> float:
            x = -np.inf
            while x < 0:
                x = np.random.normal(E, error_level * E)
            return x

        Ex = YieldPositiveEnergyDensity()
        
        # shear and bulk are gaussian distributed
        pix = np.random.normal(pi, np.fabs(error_level * pi))
        Pix = np.random.normal(Pi, np.fabs(error_level * Pi))
        
        return np.array([Ex, pix, Pix]) 

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

default_params =  {
    'tau_0':        0.1,
    'Lambda_0':     0.2 / 0.197,
    'xi_0':         -0.90, 
    'alpha_0':      0.654868759, #2 * pow(10, -3),
    'tau_f':        12.1,
    'mass':         1.015228426,
    'eta_s':        5 / (4 * np.pi),
    'pl0':          8.1705525351457684,
    'pt0':          1.9875332965147663,
    'hydro_type':   0
}

if __name__ == '__main__':
    # Flags for flow control of analysis:
    b_run_new_hydro = True     # If true, it tells HydroBayesAnalysis class to generate training points for GPs. 
    b_train_GP = True          # If true, HydroBayesAnalysis fits GPs to available training points
    b_read_in_exact = True      # If true, reads in last stored values for exact evolution. Set to false and edit parameters to change 
    b_read_mcmc = False         # If true, reads in last store MCMC chains
    b_read_observables = False  # If true, reads in the observables (E, Pi, pi) calculated using the last MCMC chains
    
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
    quit()
    if b_run_new_hydro and True:
        bayesian_analysis_class_instance.RunExactHydroForGPDesignPoints()

    exact_out = []
    # true_params = [5 / (4 * np.pi), 0.1, 0.2/.197, 2 * pow(10, -3), 0]
    true_params = [5 / (4 * np.pi)]
    if b_read_in_exact:
        with open('hydro_simulation_points/exact_output_various_times.dat','r') as f:
            exact_out = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])
    else:
        bayesian_analysis_class_instance.params['tau_f'] = 12.1
        bayesian_analysis_class_instance.params['hydro_type'] = 4
        output = bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, true_params, store_whole_file=True)
        tau_start = 0.1
        delta_tau = tau_start / 20
        observ_indices = (simulation_taus - np.full_like(simulation_taus, tau_start)) / delta_tau

        exact_out = np.array([output[int(i)-1] for i in observ_indices])

        with open('hydro_simulation_points/exact_output_various_times.dat','w') as f:
            for line in exact_out:
                for entry in line:
                    f.write(f'{entry} ')
                f.write('\n')

    alpha_error = 0.05
    exact_psuedo = np.zeros((simulation_taus.shape[0], 4))
    for i, tau in enumerate(simulation_taus):
        exact_psuedo[i, 0] = tau
        exact_psuedo[i, 1:4] = SampleObservables(alpha_error, exact_out[i, 1:4], False)

    psuedo_error = alpha_error * exact_out[:,1:4]

    bayesian_analysis_class_instance.RunMCMC(nsteps=200, nburn=50, ntemps=20, exact_observables=exact_psuedo, exact_error=psuedo_error, read_from_file=b_read_mcmc)
    mcmc_chains = bayesian_analysis_class_instance.MCMC_chains
    evidences = bayesian_analysis_class_instance.evidence

    print(mcmc_chains['ce'].shape)
    mcmc_observables = {}
    if b_read_observables:
        with open(f'pickle_files/mcmc_observables_n={len(GP_parameter_names)}.pkl', 'wb') as f:
            for name in bayesian_analysis_class_instance.hydro_names:
                mcmc_observables[name] = np.zeros((20 * len(GP_parameter_names), 200, 5))
                for i in range(mcmc_chains[name].shape[1]):
                    for j in range(mcmc_chains[name].shape[2]):
                        sys.stdout.write(f'\r({i},{j})')
                        mcmc_observables[name][i, j, :] = ConvertFromExactParametersToObservables(bayesian_analysis_class_instance, mcmc_chains[name][0,i,j,:])
            pickle.dump(mcmc_observables, f)
    else:
        with open(f'pickle_files/mcmc_observables_n={len(GP_parameter_names)}.pkl', 'rb') as f:
            mcmc_observables = pickle.load(f)

    if True:
        fig, ax = plt.subplots(nrows=1, ncols=len(GP_parameter_names), figsize=(len(GP_parameter_names) * 10,10))
        fig.patch.set_facecolor('white')
        
        if len(GP_parameter_names) == 1:
            axis_labels = [r'$\mathcal C$']
            limits = np.array([GP_parameter_ranges[0]])
            cmap = get_cmap(10, 'tab10')
            for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
                bins = np.linspace(*limits[0], 40, endpoint=True)
                ax.hist(mcmc_observables[name][:,:,0].flatten(), bins=bins, color=cmap(i), lw=2, histtype=u'step', label=name)
            costumize_axis(ax, axis_labels[0], r'Posterior')
            ax.legend(loc='upper right', fontsize=25)
            fig.tight_layout()
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists.pdf')
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
            fig.savefig(f'plots/{len(axis_labels)}_param_posterior_hists.pdf')

    outputs = {}
    bayesian_analysis_class_instance.params['tau_f'] = 12.1
    for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
        bayesian_analysis_class_instance.params['hydro_type'] = i
        map_values = [np.max(mcmc_chains[name][0,:,:,i]) for i in range(len(GP_parameter_names))]
        outputs[name] = bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, map_values, store_whole_file=True)

    #exact_params = [5 / (4 * np.pi), 0.1, 1.647204044, 0.654868759, -0.8320365099]
    bayesian_analysis_class_instance.params['hydro_type'] = 4
    exact_output = bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, true_params, store_whole_file=True)

    exact_e = exact_output[:, 1]
    exact_p = exact_output[:, 4]
    exact_pi_bar = exact_output[:, 2] / (exact_e + exact_p)
    exact_Pi_bar = exact_output[:, 3] / (exact_e + exact_p)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
    fig.patch.set_facecolor('white')
    costumize_axis(ax[0], r'$\tau$ [fm]', r'$\mathcal E / \mathcal E_0$'); ax[0].set_yscale('log')
    costumize_axis(ax[1], r'$\tau$ [fm]', r'$\pi / (\mathcal E + \mathcal P)$')
    costumize_axis(ax[2], r'$\tau$ [fm]', r'$\Pi / (\mathcal E + \mathcal P)$')

    ax[0].plot(exact_output[:, 0], exact_e / exact_e[0], lw=2, color=cmap(4), label='exact')
    ax[1].plot(exact_output[:, 0], exact_pi_bar, lw=2, color=cmap(4))
    ax[2].plot(exact_output[:, 0], exact_Pi_bar, lw=2, color=cmap(4))

    for i, name in enumerate(bayesian_analysis_class_instance.hydro_names):
        output = outputs[name]
        e = output[:,1]
        p = output[:,4]
        pi_bar = output[:,2] / (e + p)
        Pi_bar = output[:,3] / (e + p)
        ax[0].plot(output[:,0], e / e[0], lw=2, color=cmap(i), label=name)
        ax[1].plot(output[:,0], pi_bar, lw=2, color=cmap(i))
        ax[2].plot(output[:,0], Pi_bar, lw=2, color=cmap(i))
    ax[0].legend(loc='upper right', fontsize=20)
    fig.tight_layout()
    fig.savefig(f'plots/map_value_runs_n={len(GP_parameter_names)}.pdf')
