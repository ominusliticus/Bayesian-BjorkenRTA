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

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

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
    return


b_use_standard_scaler = True
b_plot_gp_error = True
b_inverse_transform = True

if __name__ == '__main__':
    def foo(x):
        return 0.1 * x ** 4 - 2.0 * x ** 2  + 10
    
    x_low = -5
    x_high = 5
    x = np.linspace(x_low, x_high, 200)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    fig.patch.set_facecolor('white')

    ax.plot(x, foo(x), lw=2, color='black', label='true')
    costumize_axis(ax, r'$x$', r'$f(x)$')
    
    # ---------------------------------------
    # --------------------------------------
    cmap = get_cmap(10,'tab10')
    scale = x_high - x_low
    bounds = np.outer(scale, (1e-2,1e2))

    for i, n_pts in enumerate(range(3,10,2)):
    #for i, n_pts in enumerate([40]):
        x_rand = np.random.uniform(x_low, x_high, n_pts)
        # x_rand = np.linspace(x_low, x_high, n_pts)
        y_rand = foo(x_rand)
       
        #if b_use_standard_scaler:
        SS = StandardScaler().fit(y_rand.reshape(-1,1))
        y_scaled = SS.transform(y_rand.reshape(-1,1))
        
        kernel = 1 * krnl.RBF(length_scale=scale, length_scale_bounds=bounds)
        GPR = gpr(kernel=kernel, n_restarts_optimizer=40)
        GPR.fit(x_rand.reshape(-1,1), y_scaled.reshape(-1,1))

        y_scaled_pred, y_scaled_err = GPR.predict(x.reshape(-1,1), return_std=True)
        if b_inverse_transform:
            y_pred = SS.inverse_transform(y_scaled_pred)
            y_stddev_plus = SS.inverse_transform(y_scaled_pred + y_scaled_err.reshape(-1,1)) - y_pred
            y_stddev_minus = y_pred - SS.inverse_transform(y_scaled_pred - y_scaled_err.reshape(-1,1))
            y_err = np.sqrt(y_stddev_plus ** 2 + y_stddev_minus ** 2).flatten()
            y_pred = y_pred.flatten()
        
        if b_plot_gp_error:
            ax.fill_between(x, y_pred + y_err, y_pred - y_err, color=cmap(i), alpha=0.3)
        else:
            y_pred = y_scaled_pred.flatten()
            y_err = y_scaled_err.flatten()

        ax.scatter(x_rand, y_rand, color=cmap(i))
        ax.plot(x, y_pred, lw=2, ls='dashed', color=cmap(i), label=f'pts = {n_pts}')
        

        #else:
        kernel = 1 * krnl.RBF(length_scale=scale, length_scale_bounds=bounds)
        GPR = gpr(kernel=kernel, n_restarts_optimizer=40, alpha=1e-8, normalize_y=True)
        GPR.fit(x_rand.reshape(-1,1), y_rand.reshape(-1,1))

        y_pred, y_err = GPR.predict(x.reshape(-1,1), return_std=True)
        y_pred = y_pred.flatten()
        y_err = y_err.flatten()

        
        ax.scatter(x_rand, y_rand, color=cmap(i))
        ax.plot(x, y_pred, lw=2, ls='dashed', color=cmap(i), label=f'pts = {n_pts}')
        if b_plot_gp_error:
            ax.fill_between(x, y_pred + y_err, y_pred - y_err, color=cmap(i), alpha=0.3)

    ax.legend()
    ax.set_ylim(-1, 15)
    
    fig.tight_layout()
    fig.savefig('debug_plots/test_func.pdf')

