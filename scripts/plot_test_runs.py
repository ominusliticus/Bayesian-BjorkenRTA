import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import rc

import warnings

warnings.filterwarnings('ignore')

plt.style.use('classic')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def get_cmap(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)


def costumize_axis(ax, x_title, y_title):
    ax.set_xlabel(x_title, fontsize=24)
    ax.set_ylabel(y_title, fontsize=24)
    ax.tick_params(axis='both', labelsize=18)
    ax.tick_params(axis='both', which='major', length=8)
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='minor', direction='in', length=4)
    return ax

# REDO EVERYTHING FROM THIS POINT ON

m = '0.200'
exact_file = f'./../output/exact/moments_of_distribution_m={m}GeV.dat'
aniso_file = f'./../output/aniso_hydro/e_m={m}GeV.dat'
altaniso_file = f'./../output/aniso_hydro/e2_m={m}GeV.dat'
ce_file = f'./../output/CE_hydro/e_m={m}GeV.dat'
dnmr_file = f'./../output/DNMR_hydro/e_m={m}GeV.dat'
aniso_pi_file = f'./../output/aniso_hydro/shear_m={m}GeV.dat'
aniso_Pi_file = f'./../output/aniso_hydro/bulk_m={m}GeV.dat'
altaniso_pi_file = f'./../output/aniso_hydro/shear2_m={m}GeV.dat'
altaniso_Pi_file = f'./../output/aniso_hydro/bulk2_m={m}GeV.dat'
ce_pi_file = f'./../output/CE_hydro/shear_m={m}GeV.dat'
ce_Pi_file = f'./../output/CE_hydro/bulk_m={m}GeV.dat'
dnmr_pi_file = f'./../output/DNMR_hydro/shear_m={m}GeV.dat'
dnmr_Pi_file = f'./../output/DNMR_hydro/bulk_m={m}GeV.dat'

with open(exact_file) as fin:
    lines = fin.readlines()
    tau      = np.array([float(line.split()[0]) for line in lines])
    exact_e  = np.array([float(line.split()[1]) for line in lines])
    exact_pl = np.array([float(line.split()[2]) for line in lines])
    exact_pt = np.array([float(line.split()[3]) for line in lines])
    exact_p  = np.array([float(line.split()[4]) for line in lines])

exact_pi = (2 / 3) * (exact_pt - exact_pl)
exact_Pi = (2 * exact_pt + exact_pl) / 3 - exact_p

aniso_e  = np.array([float(line.split()[1]) for line in open(aniso_file).readlines()])
aniso_p  = np.array([float(line.split()[2]) for line in open(aniso_file).readlines()])
aniso_xi = np.array([float(line.split()[3]) for line in open(aniso_file).readlines()])
aniso_pl = np.array([float(line.split()[1]) for line in open(aniso_pl_file).readlines()])
aniso_zL = np.array([float(line.split()[2]) for line in open(aniso_pl_file).readlines()])
aniso_pt = np.array([float(line.split()[1]) for line in open(aniso_pt_file).readlines()])
aniso_zT = np.array([float(line.split()[2]) for line in open(aniso_pt_file).readlines()])

altaniso_e  = np.array([float(line.split()[1]) for line in open(altaniso_file).readlines()])
altaniso_p  = np.array([float(line.split()[2]) for line in open(altaniso_file).readlines()])
altaniso_xi = np.array([float(line.split()[3]) for line in open(altaniso_file).readlines()])
altaniso_pl = np.array([float(line.split()[1]) for line in open(altaniso_pl_file).readlines()])
altaniso_zL = np.array([float(line.split()[2]) for line in open(altaniso_pl_file).readlines()])
altaniso_pt = np.array([float(line.split()[1]) for line in open(altaniso_pt_file).readlines()])
altaniso_zT = np.array([float(line.split()[2]) for line in open(altaniso_pt_file).readlines()])

ce_e     = np.array([float(line.split()[1]) for line in open(ce_file).readlines()])
ce_p     = np.array([float(line.split()[2]) for line in open(ce_file).readlines()])
ce_pi    = np.array([float(line.split()[1]) for line in open(ce_pi_file).readlines()])
ce_Pi    = nparray([float(line.split()[1]) for line in open(ce_Pi_file).readlines()])

dnmr_e   = np.array([float(line.split()[1]) for line in open(dnmr_file).readlines()])
dnmr_p   = np.array([float(line.split()[2]) for line in open(dnmr_file).readlines()])
dnmr_pi  = np.array([float(line.split()[1]) for line in open(dnmr_pi_file).readlines()])
dnmr_Pi  = np.array([float(line.split()[1]) for line in open(dnmr_Pi_file).readlines()])

fs=20

fig, ax = plt.subplots(3, 2, figsize=(20, 30))
fig.patch.set_facecolor('white')
ax[0, 0].plot(tau, exact_e, lw=2, color='black', label='exact')
ax[0, 0].plot(tau, aniso_e, lw=2, color='red', label='vah')
ax[0, 0].plot(tau, altaniso_e, lw=2, color='orange', label='mod vah')
ax[0, 0].plot(tau, ce_e, lw=2, color='blue', label='CE')
ax[0, 0].plot(tau, dnmr_e, lw=2, color='green', label='14-mom')
costumize_axis(ax[0, 0], r'$\tau$ [fm/c]', r'$\mathcal E(\tau)$ [fm$^{-4}$]')
ax[0, 0].set_xlim(left=.1, right=100.1)
ax[0, 0].set_xscale('log')
ax[0, 0].set_yscale('log')
ax[0, 0].legend(loc=1, fontsize=fs)
ax[0, 1].plot(tau, exact_e / exact_e - 1, lw=2, color='black')
ax[0, 1].plot(tau, aniso_e / exact_e - 1, lw=2, color='red', label=r'$\mathcal E_\mathrm{aniso} / \mathcal E_\mathrm{exact}-1$')
ax[0, 1].plot(tau, altaniso_e / exact_e - 1, lw=2, color='orange', label=r'$\mathcal E_\mathrm{alt-aniso} / \mathcal E_\mathrm{exact}-1$')
ax[0, 1].plot(tau, ce_e / exact_e - 1, lw=2, color='blue', label=r'$\mathcal E_\mathrm{CE} / \mathcal E_\mathrm{exact}-1$')
ax[0, 1].plot(tau, dnmr_e / exact_e - 1, lw=2, color='green', label=r'$\mathcal E_\mathrm{DNMR} / \mathcal E_\mathrm{exact}-1$')
costumize_axis(ax[0, 1], r'$\tau$ [fm/c]', r'$\mathcal E_\mathrm{hydro} / \mathcal E_\mathrm{exact}-1$')
ax[0, 1].set_xscale('log')
ax[0, 1].set_xlim(right=100.1)
ax[0, 1].legend(loc='center right', fontsize=fs)
ax[1, 0].plot(tau, exact_pi / (exact_e + exact_p), lw=2, color='black', label='exact')
ax[1, 0].plot(tau, aniso_pi / (aniso_e + aniso_p), lw=2, color='red', label='vah')
ax[1, 0].plot(tau, altaniso_pi / (altaniso_e + altaniso_p), lw=2, color='orange', label='mod vah')
ax[1, 0].plot(tau, ce_pi / (ce_e + ce_p), lw=2, color='blue', label='CE')
ax[1, 0].plot(tau, dnmr_pi / (dnmr_e + dnmr_p), lw=2, color='green', label='14-mom')
costumize_axis(ax[1, 0], r'$\tau$ [fm/c]', r'$\pi/\left(\mathcal E + \mathcal P_\mathrm{eq}\right)$ ')
ax[1, 0].set_xlim(left=.1, right=100.1)
#ax[1, 0].set_ylim(bottom=0.0)
ax[1, 0].set_xscale('log')
# ax[1, 0].set_yscale('log')
ax[1, 0].legend(loc=1, fontsize=fs)
ax[1, 1].plot(tau, aniso_xi, lw=2, color='red', label='vah')
ax[1, 1].plot(tau, altaniso_xi, lw=2, color='orange', label='mod vah')
ax[1, 1].set_xscale('log')
costumize_axis(ax[1, 1], r'$\tau$ fm/c', r'$\xi(\tau)$')
ax[1, 1].set_xlim(right=100.1)
ax[1, 1].legend(loc=1, fontsize=fs)
ax[2, 0].plot(tau, exact_Pi / (exact_e + exact_p), lw=2, color='black', label='exact')
ax[2, 0].plot(tau, aniso_Pi / (aniso_e + aniso_p), lw=2, color='red', label='vah')
ax[2, 0].plot(tau, altaniso_Pi / (altaniso_e + altaniso_p), lw=2, color='orange', label='mod vah')
ax[2, 0].plot(tau, ce_Pi / (ce_e + ce_p), lw=2, color='blue', label='CE')
ax[2, 0].plot(tau, dnmr_Pi / (dnmr_e + dnmr_p), lw=2, color='green', label='14-mom')
costumize_axis(ax[2, 0], r'$\tau$ [fm/c]', r'$\Pi / \left(\mathcal E +\mathcal P_\mathrm{eq}\right)$')
ax[2, 0].set_xlim(left=.1, right=100.1)
ax[2, 0].set_xscale('log')
# ax[2, 0].set_yscale('log')
ax[2, 0].legend(loc=1, fontsize=fs)
ax[2, 1].plot(tau, aniso_zT / (aniso_e + aniso_p), lw=2, color='red', label=r'vah $\bar\zeta^\perp_z$')
ax[2, 1].plot(tau, aniso_zL / (aniso_e + aniso_p), lw=2, color='red', ls='dashed', label=r'vah $\bar\zeta^L_z$')
ax[2, 1].plot(tau, altaniso_zT / (altaniso_e + altaniso_p), lw=2, color='orange', label=r'mod vah $\bar\zeta^\perp_z$')
ax[2, 1].plot(tau, altaniso_zL / (altaniso_e + altaniso_p), lw=2, color='orange', ls='dashed', label=r'mod vah $\bar\zeta^L_z$')
costumize_axis(ax[2, 1], r'$\tau$ fm/c', r'$\bar\zeta_z^{\perp,L} / (\mathcal E + \mathcal P_\mathrm{eq})$')
ax[2, 1].set_xscale('log')
ax[2, 1].set_xlim(right=100.1)
ax[2, 1].legend(loc='lower right', fontsize=fs)
fig.tight_layout()
fig.savefig('./../output/plots/comparison_200MeV.pdf')
