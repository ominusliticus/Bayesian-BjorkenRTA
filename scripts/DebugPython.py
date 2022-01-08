from platform import uname

from numpy.lib import stride_tricks

# If on WSL set temporary directory
if 'microsoft' in uname().release:
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/'

import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import poisson

from HydroBayesianAnaylsis import *

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

default_params =  {
    'tau_0':        0.1,
    'Lambda_0':     0.2 / 0.197,
    'xi_0':         -0.90, 
    'alpha_0':      2 * pow(10, -3),
    'tau_f':        12.1,
    'mass':         1.015228426,
    'eta_s':        5 / (4 * np.pi),
    'pl0':          8.1705525351457684,
    'pt0':          1.9875332965147663,
    'hydro_type':   0
}

if __name__ == '__main__':
    # Flags for flow control of analysis:
    b_run_new_hydro = False      # If true, it tells HydroBayesAnalysis class to generate training points for GPs. 
    b_train_GP = False           # If true, HydroBayesAnalysis fits GPs to available training points


    GP_parameter_names = ['eta_s','tau_0', 'Lambda_0', 'alpha_0', 'xi_0']
    GP_parameter_ranges = np.array([[1 / (4 * np.pi), 10 / (4 * np.pi)], [0.05, 0.15], [0.0, 5.0], [0.0, 1.0], [-1.0, 10.0]])
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
    bayesian_analysis_class_instance = HydroBayesianAnalysis(
        default_params=default_params,
        parameter_names=GP_parameter_names,
        parameter_ranges=GP_parameter_ranges,
        simulation_taus=simulation_taus,
        run_new_hydro=b_run_new_hydro,
        train_GP=b_train_GP
    )

    with open('design_points/design_points_n=5.dat','r') as f:
        specific_params = np.array([float(entry) for entry in f.readlines()[71].split()])
    bayesian_analysis_class_instance.params['hydro_type']=4
    print(specific_params)
    specific_hydro_output = np.array(bayesian_analysis_class_instance.ProcessHydro(GP_parameter_names, specific_params, store_whole_file=True))
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
    fig.patch.set_facecolor('white')
    
    e = specific_hydro_output[:,1]
    pi = specific_hydro_output[:,2]
    Pi = specific_hydro_output[:,3]
    p = specific_hydro_output[:,4]
    print(specific_hydro_output.shape)

    #specific_hydro_output[:,1] = e / e[0]
    #specific_hydro_output[:,2] = pi / (e + p)
    #specific_hydro_output[:,3] = Pi / (e + p)

    
    axis_labels = [r'$\mathcal E/\mathcal E_0$', r'$\pi/(\mathcal E + \mathcal P)$', r'$\Pi/(\mathcal E + \mathcal P)$']
    for i, label in enumerate(axis_labels):
        ax[i].plot(specific_hydro_output[:,0], specific_hydro_output[:,i+1], lw=3)
        costumize_axis(ax[i], r'$\tau$ MeV', label)

    fig.tight_layout()
    fig.savefig('plots/DebugPython.pdf')
