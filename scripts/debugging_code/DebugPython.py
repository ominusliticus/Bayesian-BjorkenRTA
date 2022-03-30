from platform import uname

from numpy.lib import stride_tricks

# If on WSL set temporary directory
if 'microsoft' in uname().release:
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/'

import sys
sys.path.append('..') # To import HydroBayesianAnalysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import poisson, norm

import pickle

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


hydro_names = ['ce', 'dnmr', 'vah', 'mvah']

# Plot full hydro outputs for one-parameter inference
def PlotFullOutputs_1param():
    print(os.getcwd())       
    with open(f'../design_points/design_points_n=1.dat','r') as f:
        design_points = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])

    hydro_simulations = dict((key,[]) for key in hydro_names)
    for name in hydro_names:
        for design_point in design_points:
            with open(f'../full_outputs/{name}_full_output_C={design_point}.dat','r') as f:
                temp = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])
            hydro_simulations[name].append(temp)
        hydro_simulations[name] = np.array(hydro_simulations[name])

    exact_hydro_output = []
    for j, design_point in enumerate(design_points):
        with open(f'../full_outputs/exact_hydro_C={design_point}.dat', 'r') as f:
            local = []
            for line in f.readlines():
                t, e, pl, pt, p = line.split()
                pi = 2 / 3 * (float(pt) - float(pl))
                Pi = (2 * float(pt) + float(pl)) / 3 - float(p)
                local.append([float(t), float(e), pi, Pi, float(p)])
            exact_hydro_output.append(local)
            del local

    exact_hydro_output = np.array(exact_hydro_output)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
    fig.patch.set_facecolor('white')
    cmap = get_cmap(10, 'tab10')

    rand_int = np.random.randint(0, design_points.size)
    for i in range(3):
        for j, name in enumerate(hydro_names):
            for k in range(design_points.size):
                alpha_value = 1 if k == rand_int else 0.1
                data = hydro_simulations[name][k]
                if i == 0 and k == rand_int:
                    ax[i].plot(data[:,0], data[:,i+1] / data[0, i+1], lw=2, color=cmap(j), label=name, alpha=alpha_value)
                elif i == 0:
                    ax[i].plot(data[:,0], data[:,i+1] / data[0, i+1], lw=2, color=cmap(j), alpha=alpha_value)
                else:
                    ax[i].plot(data[:,0], data[:,i+1] / (data[:,1] + data[:,4]), lw=2, color=cmap(j), alpha=alpha_value)

    for i in range(3):
        for k, design_point in enumerate(design_points):
            data = exact_hydro_output[k]
            alpha_value = 1 if k == rand_int else 0.1
            if i == 0 and k == rand_int:
                ax[i].plot(data[:,0], data[:,i+1] / data[0, i+1], lw=2, color=cmap(4), label='exact', alpha=alpha_value)
            else:
                ax[i].plot(data[:,0], data[:,i+1] / (data[:,1] + data[:,4]), lw=2, color=cmap(4), alpha=alpha_value)
    
    costumize_axis(ax[0], r'$\tau$ [fm/c]', r'$\mathcal E /\mathcal E_0$')
    costumize_axis(ax[1], r'$\tau$ [fm/c]', r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$')
    costumize_axis(ax[2], r'$\tau$ [fm/c]', r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$')
    ax[0].set_yscale('log')
    ax[0].legend(loc='upper right', fontsize=25)

    fig.tight_layout()
    fig.savefig('../plots/debug_full_simulations.pdf')


    axis_labels = [
            (r'$\tau$ [fm/c]', r'$\mathcal E /\mathcal E_0$'), 
            (r'$\tau$ [fm/c]', r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$'), 
            (r'$\tau$ [fm/c]', r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$')]
    fig2, ax2 = plt.subplots(nrows=design_points.size ,ncols=3, figsize=(6 * 3,6 * design_points.size))
    fig2.patch.set_facecolor('white')
    for i in range(3):
        for j, name in enumerate(hydro_names):
            for k, design_point in enumerate(design_points):
                data = hydro_simulations[name][k]
                if i == 0:
                    ax2[k, i].plot(data[:,0], data[:,i+1] / data[0, i+1], lw=2, color=cmap(j), label=name)
                else:
                    ax2[k, i].plot(data[:,0], data[:,i+1] / (data[:,1] + data[:,4]), lw=2, color=cmap(j))

    for i in range(3):
        for k, design_point in enumerate(design_points):
            data = exact_hydro_output[k]
            if i == 0:
                ax2[k, i].plot(data[:,0], data[:,i+1] / data[0, i+1], lw=2, color=cmap(4), label=f'exact+{k}')
                ax2[k, i].legend(loc='upper right', fontsize=20)
            else:
                ax2[k, i].plot(data[:,0], data[:,i+1] / (data[:,1] + data[:,4]), lw=2, color=cmap(4))
            costumize_axis(ax2[k, i], *axis_labels[i])
    
    for k in range(design_points.size):
        ax2[k, 0].set_yscale('log')

    fig2.tight_layout()
    fig2.savefig('../plots/debug_full_hydro_individ_compare.pdf')

def DebugGPEmulator_1param():
    with open(f'../design_points/design_points_n=1.dat','r') as f:
        design_points = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])
    sort_arr = np.argsort(design_points, 0)
    
    simulation_taus = np.array([5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
    observation_indices = np.array((simulation_taus - np.full_like(simulation_taus, 0.1)) / (0.1 / 20), dtype=int)
    simulation_points = dict((key,[]) for key in hydro_names)
    for name in hydro_names:
        for tau in simulation_taus:
            with open(f'../hydro_simulation_points/{name}_simulation_points_n=1_tau={tau}.dat','r') as f:
                temp = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])
            simulation_points[name].append(temp)
        simulation_points[name] = np.array(simulation_points[name])
    

    # After the first run, you should update files to reflect param count
    # with open(f'../pickle_files/scalers_data_n=1.pkl','rb') as f:
    #     scalers = pickle.load(f)
    with open(f'../pickle_files/emulators_data_n=1.pkl','rb') as f:
        emulators = pickle.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
    fig.patch.set_facecolor('white')
    cmap = get_cmap(10, 'tab10')
    x_vals = np.linspace(1 / (4 * np.pi), 10 / (4 * np.pi), 1000)
    for i, name in enumerate(hydro_names):
        print(name)
        data = simulation_points[name]
        for j in range(3):
            for k, tau in enumerate(simulation_taus):
                if k == 0:
                    ax[j].scatter(design_points[sort_arr[:,0]], data[sort_arr[:,0],observation_indices[k]-1,j+1], lw=2, color=cmap(i), marker='o', label=name)
                else:
                    ax[j].scatter(design_points[sort_arr[:,0]], data[sort_arr[:,0],observation_indices[k]-1,j+1], lw=2, color=cmap(i), marker='o')

                # Get emulator prediction and rescale with scalers
                emul_pred, emul_err = emulators[name][k][j].predict(x_vals.reshape(-1,1), return_std=True)
                emul_err = emul_err.reshape(-1,1)
                # pred = scalers[name][k][j].inverse_transform(emul_pred)
                # err_plus = scalers[name][k][j].inverse_transform(emul_pred + emul_err) - pred
                # err_minus = pred - scalers[name][k][j].inverse_transform(emul_pred - emul_err)
                # err = np.sqrt(err_plus * err_plus + err_minus * err_minus)

                # prepare rescaled predictions for plotting
                pred = emul_pred.reshape(-1,)
                err = emul_err.reshape(-1,)
                ax[j].fill_between(x_vals.reshape(-1,), pred + err, pred - err, color=cmap(i), alpha=0.5)
    costumize_axis(ax[0], r'$\mathcal C$', r'$\mathcal E$ [fm$^{-4}$]')
    ax[0].legend(loc='upper left', fontsize=20)
    costumize_axis(ax[1], r'$\mathcal C$', r'$\pi$ [fm$^{-4}$]')
    costumize_axis(ax[2], r'$\mathcal C$', r'$\Pi$ [fm$^{-4}$]')
    fig.tight_layout()
    fig.savefig('../plots/debug_emulator_and_scalers.pdf')
    
    return
    
    # check to see if scalers are normally distributed
    scaled_hydro = {}
    for i, name in enumerate(hydro_names):
        with open(f'../full_outputs/{name}_scaled_hydro_output_n=1.dat','r') as f:
            scaled_hydro[name] = np.array([[float(entry) for entry in line.split()] for line in f.readlines()])

        fig, ax = plt.subplots(nrows=simulation_taus.size, ncols=3, figsize=(6 * 4, 6 * simulation_taus.size))
        fig.patch.set_facecolor('white')
    for j, tau in enumerate(simulation_taus):
        for k in range(3): # observables
            for i, name in enumerate(hydro_names):
                scaler = scalers[name][j][k]
                x = np.linspace(-3 * scaler.scale_ + scaler.mean_, 3 * scaler.scale_ + scaler.mean_, 100)
                y = scaler.scale_ * scaled_hydro[name][k+j*3] + np.full_like(scaled_hydro[name][0], scaler.mean_)
                if k == 0:
                    ax[j, k].plot(x, norm.pdf(x, scaler.mean_, scaler.scale_), color=cmap(i), label=name)
                else:
                    ax[j, k].plot(x, norm.pdf(x, scaler.mean_, scaler.scale_), color=cmap(i))
                ax[j, k].scatter(y, norm.pdf(y, scaler.mean_, scaler.scale_), color=cmap(i))
                costumize_axis(ax[j, k], r'$\mathcal C$', r'z-scores')
            ax[j, 0].legend()
    fig.tight_layout()
    fig.savefig(f'../plots/debug_z-scores.pdf')

if __name__ == "__main__":
    # PlotFullOutputs_1param()
    DebugGPEmulator_1param()
