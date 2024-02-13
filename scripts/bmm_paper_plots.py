import numpy as np
import matplotlib.pyplot as plt
import my_plotting as mp
from pathlib import Path
from typing import Tuple

from hydro_code_api import HydroCodeAPI as HCA

from mpl_toolkits.axes_grid1 import make_axes_locatable


def hydro_type_from_string(hydro_name: str) -> int:
    match hydro_name:
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


def get_temp(
        energy_density: float,
        mass: float,
) -> float:
    from scipy.special import kn
    
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
        
    return mid


def extract_numbers_from_file(
    hydro_name: str,
    path_to_output: Path,
    mass: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hca = HCA(path_to_output)

    if hydro_name == 'exact':
        output = hca.get_exact_results({'mass': mass})
    else:
        output = hca.get_from_output_files(
            params_dict={'hydro_type': hydro_type_from_string(hydro_name),
                         'mass': mass},
            use_PL_PT=True,
        )

    output = output.T

    tau = output[0]
    e = output[1]
    pt = output[2]
    pl = output[3]
    p = output[4]

    pi = 2 * (pt - pl) / 3
    Pi = (2 * pt + pl) / 3 - p
    h = e + p

    return np.array([tau / tau[0], e / e[0], pi / h, Pi / h])


def return_residual(
        hydro: np.ndarray,
        kinetic: np.ndarray,
) -> np.ndarray:
    e_hydro = hydro[1]
    p_hydro = hydro[-1]
    pi_hydro = hydro[2]
    Pi_hydro = hydro[3]

    e_kinetic = kinetic[1]
    p_kinetic = kinetic[-1]
    pi_kineitc = kinetic[2]
    Pi_kineitc = kinetic[3]

    pi_reynolds_hydro = pi_hydro / (e_hydro + p_hydro)
    pi_reynolds_kinetic = pi_kineitc / (e_kinetic + p_kinetic)

    Pi_reynolds_hydro = Pi_hydro / (e_hydro + p_hydro)
    Pi_reynolds_kinetic = Pi_kineitc / (e_kinetic + p_kinetic)

    e_resid = np.fabs((e_hydro - e_kinetic) / e_kinetic)
    pi_resid = np.fabs(
        (pi_reynolds_hydro - pi_reynolds_kinetic) / pi_reynolds_kinetic
    )
    Pi_resid = np.fabs(
        (Pi_reynolds_hydro - Pi_reynolds_kinetic) / Pi_reynolds_kinetic
    )

    ret_val = np.array([e_resid, pi_resid, Pi_resid])
    ret_val[np.where(np.isnan(ret_val))] = 1.0
    return ret_val


if __name__ == "__main__":
    hydro_names = ['ce', 'dnmr', 'mis', 'vah', 'mvah', 'exact']
    colors = mp.get_cmap(10, 'tab10')
    eta_S = 0.39

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 1.2 * 5))
    fig.patch.set_facecolor('white')

    output = dict(
        (
            hydro_name,
            extract_numbers_from_file(
                hydro_name=hydro_name,
                # path_to_output='../output/magenta',
                # path_to_output='../output/maroon',
                # path_to_output='../output/blue',
                path_to_output='../output/test_no_filesystem',
                mass=0.200 / 0.197,
            )
        )
        for hydro_name in hydro_names
    )

    for i, hydro_name in enumerate(hydro_names):
        resids = return_residual(output[hydro_name], output['exact'])
        for j, y_axis in enumerate([
            r'$\mathcal E / \mathcal E_0$',
            r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$',
            r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$'
        ]):
            tau_R = np.array([
                5 * eta_S * get_temp(
                    energy_density=output[hydro_name][1],
                    mass=0.200 / 0.197
                )
            ])
            ax[j].plot(
                output[hydro_name][0],
                output[hydro_name][j + 1],
                color=colors(i)
                if hydro_name != 'exact' else 'black',
                lw=2,
                label=hydro_name.upper()
                if hydro_name != 'exact' else hydro_name,
                ls='solid'
                if hydro_name != 'exact' else 'dashed',
            )
            mp.costumize_axis(
                ax=ax[j],
                x_title='',
                y_title=y_axis
            )
            ax[j].set_xscale('log')
            ax[j].set_xlim(1, 1000)
            ax[j].set_xticks([])

    for j, y_axis in enumerate([
        r'$\mathcal E / \mathcal E_0$',
        r'$\pi / (\mathcal E + \mathcal P_\mathrm{eq})$',
        r'$\Pi / (\mathcal E + \mathcal P_\mathrm{eq})$'
    ]):
        divider = make_axes_locatable(ax[j])
        ax2 = divider.append_axes(
            "bottom",
            size="33%",
            pad=0
        )
        ax[j].figure.add_axes(ax2)

        for i, hydro_name in enumerate(hydro_names):
            if hydro_name == 'exact':
                continue
            tau_R = np.array([
                5 * eta_S * get_temp(
                    energy_density=output[hydro_name][1],
                    mass=0.200 / 0.197
                )
            ])
            resids = return_residual(output[hydro_name], output['exact'])
            ax2.plot(
                output[hydro_name][0],
                resids[j],
                color=colors(i),
                lw=2
            )
        ax2.set_xlim(1, 1000)
        mp.costumize_axis(
            ax=ax2,
            x_title=r'$\tau / \tau_R$',
            y_title=''
        ) 
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.locator_params('y', numticks=6)

    ax[0].legend(fontsize=16)
    ax[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig('./plots/hydro_compare_1.pdf')
