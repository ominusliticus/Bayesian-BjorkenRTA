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

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 1.2 * 5))
    fig.patch.set_facecolor('white')

    output = dict(
        (
            hydro_name,
            extract_numbers_from_file(
                hydro_name=hydro_name,
                # path_to_output='../output/magenta',
                path_to_output='../output/maroon',
                # path_to_output='../output/blue',
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
            resids = return_residual(output[hydro_name], output['exact'])
            ax2.plot(
                output[hydro_name][0],
                resids[j],
                color=colors(i),
                lw=2
            )
        mp.costumize_axis(
            ax=ax2,
            x_title=r'$\tau / \tau_0$',
            y_title=''
        ) 
        ax2.set_xscale('log')
        ax[j].set_xscale('log')
        ax2.set_xlim(1, 1000)
        ax[j].set_xlim(1, 1000)
        ax[j].set_xticks([])
        ax2.set_yscale('log')

    ax[0].legend(fontsize=16)
    ax[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig('./plots/hydro_compare_1.pdf')
